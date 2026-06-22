# FreeMoCap Pipeline

## Setup

```bash
conda create -n freemocap python=3.11
conda activate freemocap
cd /data1/shanmu/ai-fitness-coach/freemocap
pip install -e .
```

---

## 1. Camera Calibration

### Recommended recording procedure

Use Charuco videos, not fast waving.

For **intrinsic calibration**, record videos where the board paints the FOV for each camera:
- Move the board through the center, corners, and edges of the image.
- Include near/far distances and tilted board poses.
- Use a move-pause-move-pause pattern so the pipeline can select sharp frames.
- Keep lighting bright enough for short exposure and minimal motion blur.
- Re-run intrinsics only when the physical camera, resolution, focus, lens, or FOV changes.

For **extrinsic calibration**, keep all cameras fixed and record synchronized videos:
- Move the board through the shared capture volume.
- Pause for 0.5-1.0 seconds at each pose.
- Make sure each neighboring camera pair sees the board together at several poses.
- Avoid fast motion; synchronization error matters much less while the board is held still.

For new synced data, first run postprocessing:
```bash
python ./pose_recording/extract_sync_data_allWebcam.py --exp_name calibration_0609_1320 --camera-workers 5
```
Can also run `pose_recording/batch_postprocess.py` for processing multiple folders, e.g.
```bash
python ./pose_recording/batch_postprocess.py --pattern "calibration_0616*"
```

### Split calibration with cached intrinsics

Run both stages from one synchronized video folder:
```bash
python pose_recording/run_freemocap_split_calibration.py \
  --exp_name calibration_0603 \
  --charuco_square_size 156.6 \
  --mode split
```

Outputs:
- `cameras/intrinsics/<camera>_intrinsics.pkl`
- `cameras/intrinsics/intrinsic_calibration_summary.csv`
- `cameras/intrinsics/board_center_plots/*_intrinsic_board_centers_2d.png`
- `cameras/charuco_annotated_videos/*_charuco.mp4`
- `cameras/cameras_camera_calibration.toml`
- `cameras/extrinsic_calibration_summary.csv`
- `cameras/board_center_plots/*_extrinsic_board_centers_2d.png`
- `cameras/cameras_camera_calibration_3d.png`

The console prints, for each camera, how many sampled frames contained a board, how many sharp frames were selected, and how long the calibration solve took. Camera-video scanning runs in parallel by default; in `--mode split`, the same detections are reused for the extrinsic stage instead of scanning the videos a second time. The 2D board-center plots show whether selected frames cover each camera image or cluster in one region. Annotated Charuco videos are saved by default so missed detections can be inspected visually.

Run only intrinsics:
```bash
python pose_recording/run_freemocap_split_calibration.py \
  --exp_name calibration_intrinsics_0618 \
  --charuco_square_size 156.6 \
  --max-intrinsic-frames-per-camera 300 \
  --mode intrinsics
```

Rerun intrinsics for one camera after an interrupted/slow solve:
```bash
python pose_recording/run_freemocap_split_calibration.py \
  --exp_name calibration_intrinsics_0618 \
  --charuco_square_size 156.6 \
  --mode intrinsics \
  --cameras cam4_synced \
  --max-intrinsic-frames-per-camera 300 \
  --max-outlier-passes 3
```

If the terminal stops at `solve pass ...`, it is inside OpenCV `cv2.calibrateCamera`. That call has no progress callback. Large solves such as 400-500 frames with 2-3 outlier passes can look frozen; use 80-120 diverse frames for a fast intrinsic rerun, then only increase the cap for a final quality check if needed.

Run only extrinsics using existing intrinsics:
```bash
python pose_recording/run_freemocap_split_calibration.py \
  --exp_name extrinsics_01 \
  --charuco_square_size 156.6 \
  --mode extrinsics \
  --max-frames-per-camera 500
  --intrinsics-dir ssd_datas/fitness_data/synchronized/intrinsics_new_4/cameras/intrinsics
```

Useful selection arguments:
```bash
--detector freemocap         # use FreeMoCap/skellytracker cv2.aruco.CharucoDetector
--frame-step 2                # sample every Nth video frame
--sharpness-quantile 0.2      # candidate frames must be at/above this sharpness quantile
--selection-strategy diverse  # diverse=spread board centers/scale, sharpest=old behavior
--selection-grid-cols 4       # image-space diversity grid columns
--selection-grid-rows 3       # image-space diversity grid rows
--selection-scale-bins 2      # near/far board-size diversity bins
--max-intrinsic-frames-per-camera 120 # cap per-camera intrinsic solve frames
--max-frames-per-camera 250   # cap synchronized extrinsic poses
--min-frame-gap 5             # avoid selecting many adjacent duplicate frames
--min-corners 4               # minimum corners for detection/annotation
--min-calibration-corners 6   # minimum corners for cv2 intrinsic calibration
--max-outlier-passes 1        # 1=fast default; use 2-3 for final outlier refinement
--progress-interval 300       # print scan progress every N sampled frames
--num-detection-workers 0     # 0=auto parallel scan, 1=serial
--num-intrinsic-workers 0     # 0=auto parallel intrinsic solves, 1=serial
--no-detection-videos          # skip annotated Charuco videos if runtime/storage matters
--cameras cam4_synced        # only process specific camera(s), useful for intrinsic reruns
--plot-path path/to/layout.png # optional output path for the 3D camera layout
```

Detection defaults now mirror the current FreeMoCap/skellytracker Charuco path: `cv2.aruco.CharucoDetector.detectBoard`, accepting frames with at least 4 Charuco corners. Frame selection first filters detections by sharpness. With the default `diverse` strategy, the sharp candidates are bucketed by board center across the image and by board apparent size, then selected round-robin from those buckets while respecting `--min-frame-gap`. This gives better FOV coverage than simply taking the sharpest frames. Use `--selection-strategy sharpest` to recover the old behavior.

### Legacy one-shot FreeMoCap calibration

The older combined intrinsic+extrinsic solve is still available:
```bash
python pose_recording/run_freemocap_calibration.py --charuco_square_size 156.6 --exp_name calibration_0603
```

Optional arguments:
```bash
--charuco_square_size 156.6   # black square side length in mm (default: 156.6)
--charuco_board 5x3           # board type: 5x3 (default) or default
--conda_env freemocap         # conda env name (default: freemocap)
```

Output: `cameras/cameras_camera_calibration.toml`


### Assessing calibration quality

Look for `error: X.XX` in the console output during calibration.

| Reprojection error | Quality |
|--------------------|---------|
| < 0.5 px | Excellent |
| 0.5 - 1.0 px | Good |
| 1.0 - 2.0 px | Acceptable |
| > 2.0 px | Poor - redo calibration |

In the output `.toml`, sanity-check each camera:
- **Focal length** (`matrix[0][0]`): should be consistent across cameras of the same model (~400-600 px for wide-angle at 1280x720). Values in the thousands indicate a failed camera.
- **Distortion** (`distortions[0]`): should be < 1.0. Values > 10 indicate a failed camera.
- **World position**: all cameras should be within a physically plausible range (< +/-5000 mm for a 2-5 m capture volume).

---

## 2. 3D Skeleton Extraction
Session must have `cameras/cam*_synced.mp4` and a `config.yaml` with `freemocap.calibration_path`

Also first run the following command:
```bash
python ./pose_recording/extract_sync_data_allWebcam.py --exp_name calibration_0609_1320 --camera-workers 5
```

Then run:
```bash
conda run --no-capture-output -n freemocap python skeleton-preprocessing/skeleton-preprocessing/preprocess.py   --freemocap_session ssd_datas/fitness_data/synchronized/movement_0622   --freemocap_calibration_toml /data1/shanmu/ai-fitness-coach/ssd_datas/fitness_data/synchronized/extrinsics_long_02/cameras/cameras_camera_calibration.toml   --freemocap_run_vitpose   --freemocap_skeleton_3d   --freemocap_smooth_skeleton_3d   --freemocap_vitpose_gpu_ids 0,1,2,3  --freemocap_vitpose_camera_workers 4   --overwrite
```


Run this for visualization:
```bash
python pose_recording/visualize_skeleton_json_3d.py ssd_datas/fitness_data/synchronized/movement_0622/cameras/skeleton_preprocessing/skeleton_smoothed.json ssd_datas/fitness_data/synchronized/movement_0622/cameras/skeleton_preprocessing/skeleton.mp4 --backend matplotlib
```

Run this to visualize 3D reprojection back onto the synced camera videos:
```bash
python pose_recording/visualize_freemocap_reprojection.py \
  ssd_datas/fitness_data/synchronized/movement_0622 \
  --calibration-toml ssd_datas/fitness_data/synchronized/extrinsics_long_02/cameras/cameras_camera_calibration.toml
```

This writes per-camera overlay videos to
`ssd_datas/fitness_data/synchronized/movement_0622/cameras/skeleton_preprocessing/reprojection_videos/`.
The overlay uses the processed per-camera calibration `.pkl` files when present,
projects the 3D skeleton onto raw video pixels, compares against
`vitpose_raw_json/cam*_vitpose_keypoints.json` when available, and then writes
`reprojection_grid_3x2_raw_skeleton_smoothed.mp4`.

Standalone ffmpeg command for the same 3x2 grid:
```bash
ffmpeg -hide_banner -y \
  -i ssd_datas/fitness_data/synchronized/movement_0622/cameras/skeleton_preprocessing/reprojection_videos/cam0_reprojection_raw_skeleton_smoothed.mp4 \
  -i ssd_datas/fitness_data/synchronized/movement_0622/cameras/skeleton_preprocessing/reprojection_videos/cam1_reprojection_raw_skeleton_smoothed.mp4 \
  -i ssd_datas/fitness_data/synchronized/movement_0622/cameras/skeleton_preprocessing/reprojection_videos/cam2_reprojection_raw_skeleton_smoothed.mp4 \
  -i ssd_datas/fitness_data/synchronized/movement_0622/cameras/skeleton_preprocessing/reprojection_videos/cam3_reprojection_raw_skeleton_smoothed.mp4 \
  -i ssd_datas/fitness_data/synchronized/movement_0622/cameras/skeleton_preprocessing/reprojection_videos/cam4_reprojection_raw_skeleton_smoothed.mp4 \
  -f lavfi -t 48.9 -i color=c=black:s=1280x720:r=30 \
  -filter_complex "[0:v]scale=1280:720,setsar=1,setpts=PTS-STARTPTS[v0];[1:v]scale=1280:720,setsar=1,setpts=PTS-STARTPTS[v1];[2:v]scale=1280:720,setsar=1,setpts=PTS-STARTPTS[v2];[3:v]scale=1280:720,setsar=1,setpts=PTS-STARTPTS[v3];[4:v]scale=1280:720,setsar=1,setpts=PTS-STARTPTS[v4];[5:v]scale=1280:720,setsar=1,setpts=PTS-STARTPTS[v5];[v0][v1][v2][v3][v4][v5]xstack=inputs=6:layout=0_0|1280_0|2560_0|0_720|1280_720|2560_720[v]" \
  -map "[v]" -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p -shortest \
  ssd_datas/fitness_data/synchronized/movement_0622/cameras/skeleton_preprocessing/reprojection_videos/reprojection_grid_3x2_raw_skeleton_smoothed.mp4
```

---

## 3. Reprojection Error Analysis (Legacy)

After extraction, analyze 3D quality with:

```bash
python pose_recording/analyze_reprojection_error.py --exp_name session_test
```

Prints a summary report and saves 6 plots to
`cameras/output_data/reprojection_error_analysis/`:

| Plot | What it shows |
|------|---------------|
| `01_frame_error_over_time.png` | Median error per frame; reveals when the subject left the capture volume |
| `02_per_camera_boxplot.png` | Error distribution per camera; flags bad cameras |
| `03_body_landmark_error.png` | Per-landmark bar chart for the 33 body keypoints |
| `04_body_error_heatmap.png` | Frame × landmark heatmap; bright = high error |
| `05_error_cdf_by_segment.png` | CDF curves for body / hands / face separately |
| `06_per_camera_frame_heatmap.png` | Camera × frame heatmap; shows if a camera fails only in certain windows |

**Quality thresholds (median reprojection error):**

| Error | Quality |
|-------|---------|
| < 5 px | Excellent |
| 5 – 15 px | Good |
| 15 – 30 px | Acceptable |
| > 30 px | Poor — check calibration and camera coverage |

Use **median**, not mean — a single out-of-view frame inflates the mean to millions of pixels.

---

## 4. Visualization

### 3D skeleton video

```bash
python ./pose_recording/visualize_mediapipe_3d.py pose_recording/mediapipe_body_3d_xyz.csv --output ./pose_recording/test.mp4
```
(consider using py39)

### Combined annotated camera views (3×2 grid)
```bash
ffmpeg \
  -i cam0_synced.mp4 \
  -i cam1_synced.mp4 \
  -i cam2_synced.mp4 \
  -i cam3_synced.mp4 \
  -i cam4_synced.mp4 \
  -filter_complex "
[0:v]drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='Cam 0':x=20:y=20:fontsize=36:fontcolor=white:box=1:boxcolor=black@0.55[v0];
[1:v]drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='Cam 1':x=20:y=20:fontsize=36:fontcolor=white:box=1:boxcolor=black@0.55[v1];
[2:v]drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='Cam 2':x=20:y=20:fontsize=36:fontcolor=white:box=1:boxcolor=black@0.55[v2];
[3:v]drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='Cam 3':x=20:y=20:fontsize=36:fontcolor=white:box=1:boxcolor=black@0.55[v3];
[4:v]drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='Cam 4':x=20:y=20:fontsize=36:fontcolor=white:box=1:boxcolor=black@0.55[v4];
[v0][v1][v2][v3][v4]xstack=inputs=5:layout=0_0|w0_0|w0+w1_0|0_h0|w3_h0[v]
" \
  -map "[v]" \
  -c:v libx264 -crf 18 -preset medium \
  -pix_fmt yuv420p \
  -shortest \
  combined_output_3x2_5cams.mp4
```


```bash
ffmpeg \
  -i cam0_synced.mp4 \
  -i cam1_synced.mp4 \
  -i cam2_synced.mp4 \
  -i cam3_synced.mp4 \
  -i cam4_synced.mp4 \
  -filter_complex "
[0:v]drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='Cam 0':x=20:y=20:fontsize=36:fontcolor=white:box=1:boxcolor=black@0.55[v0];
[1:v]drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='Cam 1':x=20:y=20:fontsize=36:fontcolor=white:box=1:boxcolor=black@0.55[v1];
[2:v]drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='Cam 2':x=20:y=20:fontsize=36:fontcolor=white:box=1:boxcolor=black@0.55[v2];
[3:v]drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='Cam 3':x=20:y=20:fontsize=36:fontcolor=white:box=1:boxcolor=black@0.55[v3];
[4:v]drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='Cam 4':x=20:y=20:fontsize=36:fontcolor=white:box=1:boxcolor=black@0.55[v4];
[v0][v1][v2][v3][v4]xstack=inputs=5:layout=0_0|0_h0|w0_0|w0_h0|w0+w2_0[v]
" \
  -map "[v]" \
  -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p \
  combined_output_5cams.mp4
```

### 3D skeleton + camera grid side by side

```bash
ffmpeg -y \
  -i skeleton.mp4 \
  -i combined_output_5cams.mp4 \
  -filter_complex "
[0:v]pad=iw:800:0:(oh-ih)/2:black,setsar=1[v0];
[1:v]scale=-2:800,setsar=1[v1];
[v0][v1]hstack=inputs=2[v]
" \
  -map "[v]" \
  -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p \
  -shortest \
  output.mp4
```
