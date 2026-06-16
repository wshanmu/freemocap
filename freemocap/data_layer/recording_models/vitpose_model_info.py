from pathlib import Path

from skellytracker.trackers.base_tracker.base_tracking_params import BaseTrackingParams
from skellytracker.trackers.yolo_tracker.yolo_model_info import YOLOModelInfo


VITPOSE_ROOT = Path(__file__).resolve().parents[3] / "vitpose-deployment"
VITPOSE_CHECKPOINTS = VITPOSE_ROOT / "checkpoints"
VITPOSE_COCO_CONFIG_DIR = (
    VITPOSE_ROOT
    / "configs"
    / "body"
    / "2d_kpt_sview_rgb_img"
    / "topdown_heatmap"
    / "coco"
)


class VitPoseModelInfo(YOLOModelInfo):
    name = "vitpose"
    tracker_name = "ViTPoseTracker"


class VitPoseTrackingParams(BaseTrackingParams):
    vitpose_root: str = str(VITPOSE_ROOT)
    vitpose_env: str = "vitpose"
    pose_config: str = str(VITPOSE_COCO_CONFIG_DIR / "ViTPose_large_coco_256x192.py")
    pose_checkpoint: str = str(VITPOSE_CHECKPOINTS / "vitpose_large.pth")
    yolo_model: str = str(VITPOSE_CHECKPOINTS / "yolo11s.pt")
    yolo_env: str = "freemocap"
    bbox_threshold: float = 0.7
    keypoint_threshold: float = 0.7
    device: str = "cuda:0"
    yolo_device: str = "0"
    gpu_ids: str = ""
    camera_worker_count: int = 0
    yolo_imgsz: int = 640
    yolo_batch_size: int = 16
    yolo_fp32: bool = False
    no_flip_test: bool = True
    keep_intermediate_json: bool = False
    save_annotated_videos: bool = True
