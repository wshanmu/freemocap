import json
import logging
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from freemocap.data_layer.recording_models.vitpose_model_info import (
    VitPoseTrackingParams,
)
from freemocap.utilities.get_video_paths import get_video_paths


logger = logging.getLogger(__name__)


COCO_SKELETON_CONNECTIONS = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
]

COCO_KEYPOINT_COLORS = [
    (0, 255, 255),
    (0, 191, 255),
    (0, 191, 255),
    (0, 127, 255),
    (0, 127, 255),
    (0, 255, 0),
    (0, 255, 0),
    (80, 220, 80),
    (80, 220, 80),
    (120, 200, 120),
    (120, 200, 120),
    (255, 140, 0),
    (255, 140, 0),
    (255, 90, 0),
    (255, 90, 0),
    (255, 40, 0),
    (255, 40, 0),
]


@dataclass(frozen=True)
class DeviceAssignment:
    label: str
    vitpose_device: str
    yolo_device: str
    cuda_visible_devices: str | None = None


def _extract_video_frames(video_path: Path, frame_dir: Path) -> int:
    frame_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_index = 0
    success, frame = capture.read()
    while success:
        frame_path = frame_dir / f"frame_{frame_index:06d}.jpg"
        if not cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Could not write frame image: {frame_path}")
        frame_index += 1
        success, frame = capture.read()

    capture.release()
    if frame_index == 0:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    return frame_index


def _person_sort_key(person: dict) -> tuple[float, float]:
    bbox = person.get("bbox") or []
    score = float(bbox[4]) if len(bbox) >= 5 else 0.0
    if len(bbox) >= 4:
        area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
    else:
        area = 0.0
    return score, area


def _load_vitpose_json(
    json_path: Path,
    expected_frames: int,
    keypoint_threshold: float,
    num_tracked_points: int,
) -> np.ndarray:
    records = json.loads(json_path.read_text(encoding="utf-8"))
    if len(records) != expected_frames:
        raise RuntimeError(
            f"ViTPose returned {len(records)} frame records for {json_path}, expected {expected_frames}"
        )

    data = np.full((expected_frames, num_tracked_points, 3), np.nan, dtype=float)
    for frame_index, record in enumerate(records):
        people = record.get("people") or []
        if not people:
            continue

        person = max(people, key=_person_sort_key)
        for keypoint in person.get("keypoints", []):
            keypoint_id = int(keypoint["id"])
            if keypoint_id >= num_tracked_points:
                continue

            score = float(keypoint["score"])
            data[frame_index, keypoint_id, 2] = score
            if score < keypoint_threshold:
                continue

            data[frame_index, keypoint_id, 0] = float(keypoint["x"])
            data[frame_index, keypoint_id, 1] = float(keypoint["y"])

    return data


def _run_vitpose_for_camera(
    frame_dir: Path,
    output_json_path: Path,
    tracking_params: VitPoseTrackingParams,
    device_assignment: DeviceAssignment,
) -> None:
    conda_exe = shutil.which("conda")
    if conda_exe is None:
        raise FileNotFoundError("Could not find 'conda' on PATH.")

    vitpose_root = Path(tracking_params.vitpose_root).expanduser().resolve()
    infer_script = vitpose_root / "tools" / "vitpose_2d_infer.py"
    if not infer_script.exists():
        raise FileNotFoundError(f"Missing ViTPose inference script: {infer_script}")

    env = os.environ.copy()
    if device_assignment.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = device_assignment.cuda_visible_devices

    command = [
        conda_exe,
        "run",
        "--no-capture-output",
        "-n",
        tracking_params.vitpose_env,
        "python",
        str(infer_script),
        "--pose-config",
        str(Path(tracking_params.pose_config).expanduser().resolve()),
        "--pose-checkpoint",
        str(Path(tracking_params.pose_checkpoint).expanduser().resolve()),
        "--yolo11-model",
        str(Path(tracking_params.yolo_model).expanduser()),
        "--yolo11-env",
        tracking_params.yolo_env,
        "--yolo11-device",
        device_assignment.yolo_device,
        "--yolo11-imgsz",
        str(tracking_params.yolo_imgsz),
        "--yolo11-batch-size",
        str(tracking_params.yolo_batch_size),
        "--img-root",
        str(frame_dir),
        "--out-json",
        str(output_json_path),
        "--bbox-thr",
        str(tracking_params.bbox_threshold),
        "--kpt-thr",
        str(tracking_params.keypoint_threshold),
        "--device",
        device_assignment.vitpose_device,
    ]
    if tracking_params.yolo_fp32:
        command.append("--yolo11-fp32")
    if tracking_params.no_flip_test:
        command.append("--no-flip-test")

    print(
        "[ViTPose] "
        f"{frame_dir.name}: assigned={device_assignment.label} "
        f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '<inherited>')} "
        f"vitpose_device={device_assignment.vitpose_device} "
        f"yolo_device={device_assignment.yolo_device}",
        flush=True,
    )
    logger.info("Running ViTPose command: %s", " ".join(command))
    subprocess.run(command, cwd=vitpose_root, env=env, check=True)


def _parse_vitpose_device_assignments(
    tracking_params: VitPoseTrackingParams,
) -> list[DeviceAssignment]:
    gpu_ids = [
        gpu_id.strip()
        for gpu_id in (getattr(tracking_params, "gpu_ids", "") or "").split(",")
        if gpu_id.strip()
    ]
    if not gpu_ids:
        return [
            DeviceAssignment(
                label=f"{tracking_params.device}/yolo:{tracking_params.yolo_device}",
                vitpose_device=tracking_params.device,
                yolo_device=tracking_params.yolo_device,
            )
        ]

    device_assignments = []
    for gpu_id in gpu_ids:
        if gpu_id.lower() == "cpu":
            device_assignments.append(
                DeviceAssignment(label="cpu", vitpose_device="cpu", yolo_device="cpu")
            )
        elif gpu_id.startswith("cuda:"):
            physical_gpu_id = gpu_id.split(":", maxsplit=1)[1]
            device_assignments.append(
                DeviceAssignment(
                    label=gpu_id,
                    vitpose_device="cuda:0",
                    yolo_device="0",
                    cuda_visible_devices=physical_gpu_id,
                )
            )
        else:
            device_assignments.append(
                DeviceAssignment(
                    label=f"cuda:{gpu_id}",
                    vitpose_device="cuda:0",
                    yolo_device="0",
                    cuda_visible_devices=gpu_id,
                )
            )
    return device_assignments


def _resolve_camera_worker_count(
    tracking_params: VitPoseTrackingParams,
    device_assignments: list[DeviceAssignment],
    number_of_cameras: int,
) -> int:
    requested_workers = int(getattr(tracking_params, "camera_worker_count", 0) or 0)
    worker_count = requested_workers if requested_workers > 0 else len(device_assignments)
    return max(1, min(worker_count, number_of_cameras))


def _open_video_writer(
    output_path: Path,
    fps: float,
    frame_size: tuple[int, int],
) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for codec in ("mp4v", "avc1", "H264"):
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            frame_size,
        )
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Could not open VideoWriter for {output_path}")


def _draw_vitpose_skeleton_on_frame(
    frame: np.ndarray,
    keypoints_xy_conf: np.ndarray,
    keypoint_threshold: float,
) -> np.ndarray:
    for start_index, end_index in COCO_SKELETON_CONNECTIONS:
        if start_index >= keypoints_xy_conf.shape[0] or end_index >= keypoints_xy_conf.shape[0]:
            continue

        start = keypoints_xy_conf[start_index]
        end = keypoints_xy_conf[end_index]
        if (
            np.isfinite(start[0])
            and np.isfinite(start[1])
            and np.isfinite(end[0])
            and np.isfinite(end[1])
            and np.isfinite(start[2])
            and np.isfinite(end[2])
            and start[2] >= keypoint_threshold
            and end[2] >= keypoint_threshold
        ):
            cv2.line(
                frame,
                (int(round(start[0])), int(round(start[1]))),
                (int(round(end[0])), int(round(end[1]))),
                (0, 255, 180),
                3,
                lineType=cv2.LINE_AA,
            )

    for keypoint_index, keypoint in enumerate(keypoints_xy_conf):
        if (
            not np.isfinite(keypoint[0])
            or not np.isfinite(keypoint[1])
            or not np.isfinite(keypoint[2])
            or keypoint[2] < keypoint_threshold
        ):
            continue

        color = COCO_KEYPOINT_COLORS[keypoint_index % len(COCO_KEYPOINT_COLORS)]
        center = (int(round(keypoint[0])), int(round(keypoint[1])))
        cv2.circle(frame, center, 5, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(frame, center, 7, (15, 15, 15), thickness=1, lineType=cv2.LINE_AA)

    cv2.putText(
        frame,
        "ViTPose",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    return frame


def _save_vitpose_annotated_video(
    camera_index: int,
    video_path: Path,
    camera_array: np.ndarray,
    annotated_video_folder_path: Path,
    keypoint_threshold: float,
) -> Path:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video for ViTPose annotation: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Could not read video dimensions for {video_path}")

    output_path = annotated_video_folder_path / f"{video_path.stem}_vitpose.mp4"
    writer = _open_video_writer(output_path=output_path, fps=fps, frame_size=(width, height))

    frame_index = 0
    success, frame = capture.read()
    while success:
        if frame_index < camera_array.shape[0]:
            frame = _draw_vitpose_skeleton_on_frame(
                frame=frame,
                keypoints_xy_conf=camera_array[frame_index],
                keypoint_threshold=keypoint_threshold,
            )
        writer.write(frame)
        frame_index += 1
        success, frame = capture.read()

    capture.release()
    writer.release()
    logger.info(
        "Saved ViTPose annotated video for camera %s to %s",
        camera_index,
        output_path,
    )
    return output_path


def _save_vitpose_annotated_videos(
    video_paths: list[Path],
    camera_arrays: list[np.ndarray],
    synchronized_videos_folder_path: Path,
    tracking_params: VitPoseTrackingParams,
    camera_worker_count: int,
) -> None:
    annotated_video_folder_path = synchronized_videos_folder_path.parent / "vitpose_annotated_videos"
    annotated_video_folder_path.mkdir(parents=True, exist_ok=True)
    print(f"[ViTPose] Saving annotated videos to {annotated_video_folder_path}", flush=True)

    worker_count = max(1, min(camera_worker_count, len(video_paths)))
    if worker_count == 1:
        for camera_index, video_path in enumerate(video_paths):
            _save_vitpose_annotated_video(
                camera_index=camera_index,
                video_path=video_path,
                camera_array=camera_arrays[camera_index],
                annotated_video_folder_path=annotated_video_folder_path,
                keypoint_threshold=tracking_params.keypoint_threshold,
            )
        return

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _save_vitpose_annotated_video,
                camera_index,
                video_path,
                camera_arrays[camera_index],
                annotated_video_folder_path,
                tracking_params.keypoint_threshold,
            )
            for camera_index, video_path in enumerate(video_paths)
        ]
        for future in as_completed(futures):
            future.result()


def _process_camera_video(
    camera_index: int,
    video_path: Path,
    temp_dir: Path,
    output_data_folder_path: Path,
    tracking_params: VitPoseTrackingParams,
    num_tracked_points: int,
    device_assignment: DeviceAssignment,
) -> tuple[int, np.ndarray]:
    frame_dir = temp_dir / f"cam{camera_index:02d}_frames"
    output_json_path = temp_dir / f"cam{camera_index:02d}_vitpose.json"

    logger.info(
        "Processing camera %s with device assignment %s",
        camera_index,
        device_assignment,
    )
    frame_count = _extract_video_frames(video_path=video_path, frame_dir=frame_dir)
    _run_vitpose_for_camera(
        frame_dir=frame_dir,
        output_json_path=output_json_path,
        tracking_params=tracking_params,
        device_assignment=device_assignment,
    )
    camera_array = _load_vitpose_json(
        json_path=output_json_path,
        expected_frames=frame_count,
        keypoint_threshold=tracking_params.keypoint_threshold,
        num_tracked_points=num_tracked_points,
    )

    if tracking_params.keep_intermediate_json:
        debug_json_path = output_data_folder_path / f"vitpose_cam{camera_index:02d}_keypoints.json"
        shutil.copy2(output_json_path, debug_json_path)

    return camera_index, camera_array


def run_vitpose_image_tracking(
    tracking_params: VitPoseTrackingParams,
    synchronized_videos_folder_path: Path,
    output_data_folder_path: Path,
    num_tracked_points: int,
) -> np.ndarray:
    output_data_folder_path.mkdir(parents=True, exist_ok=True)
    video_paths = get_video_paths(synchronized_videos_folder_path)
    if not video_paths:
        raise FileNotFoundError(f"No videos found in {synchronized_videos_folder_path}")

    device_assignments = _parse_vitpose_device_assignments(tracking_params)
    camera_worker_count = _resolve_camera_worker_count(
        tracking_params=tracking_params,
        device_assignments=device_assignments,
        number_of_cameras=len(video_paths),
    )
    logger.info(
        "Running ViTPose on %s camera(s) with %s worker(s) across devices: %s",
        len(video_paths),
        camera_worker_count,
        ", ".join(device_assignment.label for device_assignment in device_assignments),
    )
    print(
        "[ViTPose] "
        f"Running {len(video_paths)} camera(s) with {camera_worker_count} worker(s): "
        + ", ".join(
            f"{device_assignment.label}"
            f"->visible={device_assignment.cuda_visible_devices or '<inherited>'}"
            for device_assignment in device_assignments
        ),
        flush=True,
    )

    camera_arrays = [None] * len(video_paths)
    with tempfile.TemporaryDirectory(prefix="freemocap_vitpose_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        if camera_worker_count == 1:
            for camera_index, video_path in enumerate(video_paths):
                result_index, camera_array = _process_camera_video(
                    camera_index=camera_index,
                    video_path=Path(video_path),
                    temp_dir=temp_dir,
                    output_data_folder_path=output_data_folder_path,
                    tracking_params=tracking_params,
                    num_tracked_points=num_tracked_points,
                    device_assignment=device_assignments[camera_index % len(device_assignments)],
                )
                camera_arrays[result_index] = camera_array
        else:
            with ThreadPoolExecutor(max_workers=camera_worker_count) as executor:
                futures = [
                    executor.submit(
                        _process_camera_video,
                        camera_index,
                        Path(video_path),
                        temp_dir,
                        output_data_folder_path,
                        tracking_params,
                        num_tracked_points,
                        device_assignments[camera_index % len(device_assignments)],
                    )
                    for camera_index, video_path in enumerate(video_paths)
                ]
                for future in as_completed(futures):
                    result_index, camera_array = future.result()
                    camera_arrays[result_index] = camera_array

    missing_cameras = [
        camera_index
        for camera_index, camera_array in enumerate(camera_arrays)
        if camera_array is None
    ]
    if missing_cameras:
        raise RuntimeError(f"ViTPose did not return results for cameras: {missing_cameras}")

    frame_counts = {camera_array.shape[0] for camera_array in camera_arrays}
    if len(frame_counts) != 1:
        raise RuntimeError(
            "ViTPose camera results have different frame counts: "
            + ", ".join(
                f"cam{camera_index}={camera_array.shape[0]}"
                for camera_index, camera_array in enumerate(camera_arrays)
            )
        )

    image_data = np.stack(camera_arrays, axis=0)
    output_npy_path = output_data_folder_path / "vitpose_2dData_numCams_numFrames_numTrackedPoints_pixelXY.npy"
    confidence_npy_path = (
        output_data_folder_path / "vitpose_2dData_numCams_numFrames_numTrackedPoints_confidence.npy"
    )
    np.save(output_npy_path, image_data)
    np.save(confidence_npy_path, image_data[..., 2])
    logger.info("Saved ViTPose 2D data to %s", output_npy_path)

    if getattr(tracking_params, "save_annotated_videos", True):
        _save_vitpose_annotated_videos(
            video_paths=[Path(video_path) for video_path in video_paths],
            camera_arrays=camera_arrays,
            synchronized_videos_folder_path=synchronized_videos_folder_path,
            tracking_params=tracking_params,
            camera_worker_count=camera_worker_count,
        )

    return image_data
