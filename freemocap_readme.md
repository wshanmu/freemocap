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
for new synced data, first run for postprocessing:
```bash
python ./pose_recording/extract_sync_data_allWebcam.py --exp_name calibration_0609_1320 --camera-workers 5
```
Can also run `pose_recording/batch_postprocess.py` for processing multiple folders (remember to update the `scripts_to_run`), e.g.,
```bash
python ./pose_recording/batch_postprocess.py --pattern "calibration_0616*"
```

Then run the calibration:
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
| 0.5 – 1.0 px | Good |
| 1.0 – 2.0 px | Acceptable |
| > 2.0 px | Poor — redo calibration |

In the output `.toml`, sanity-check each camera:
- **Focal length** (`matrix[0][0]`): should be consistent across cameras of the same model (~400–600 px for wide-angle at 1280×720). Values in the thousands indicate a failed camera.
- **Distortion** (`distortions[0]`): should be < 1.0. Values > 10 indicate a failed camera.
- **World position**: all cameras should be within a physically plausible range (< ±5000 mm for a 2–5 m capture volume).

---

## 2. 3D Skeleton Extraction
Session must have `cameras/cam*_synced.mp4` and a `config.yaml` with `freemocap.calibration_path`

Also first run the following command:
```bash
python ./pose_recording/extract_sync_data_allWebcam.py --exp_name calibration_0609_1320 --camera-workers 5
```

Then run:
```bash
conda run --no-capture-output -n freemocap python skeleton-preprocessing/skeleton-preprocessing/preprocess.py   --freemocap_session ssd_datas/fitness_data/synchronized/movement_0616   --freemocap_calibration_toml /data1/shanmu/ai-fitness-coach/ssd_datas/fitness_data/synchronized/calibration_0616_01/cameras/cameras_camera_calibration.toml   --freemocap_run_vitpose   --freemocap_skeleton_3d   --freemocap_smooth_skeleton_3d   --freemocap_vitpose_gpu_ids 0,1,2,3  --freemocap_vitpose_camera_workers 4   --overwrite
```


Run this for visualization:
```bash
python pose_recording/visualize_skeleton_json_3d.py ssd_datas/fitness_data/synchronized/movement_0616/cameras/skeleton_preprocessing/skeleton_smoothed.json ssd_datas/fitness_data/synchronized/movement_0616/cameras/skeleton_preprocessing/skeleton.mp4 --backend matplotlib
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
  -i cam0_synced_yolo.mp4 \
  -i cam1_synced_yolo.mp4 \
  -i cam2_synced_yolo.mp4 \
  -i cam3_synced_yolo.mp4 \
  -i cam4_synced_yolo.mp4 \
  -i cam5_synced_yolo.mp4 \
  -filter_complex "
[0:v]drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='Cam 0':x=20:y=20:fontsize=36:fontcolor=white:box=1:boxcolor=black@0.55[v0];
[1:v]drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='Cam 1':x=20:y=20:fontsize=36:fontcolor=white:box=1:boxcolor=black@0.55[v1];
[2:v]drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='Cam 2':x=20:y=20:fontsize=36:fontcolor=white:box=1:boxcolor=black@0.55[v2];
[3:v]drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='Cam 3':x=20:y=20:fontsize=36:fontcolor=white:box=1:boxcolor=black@0.55[v3];
[4:v]drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='Cam 4':x=20:y=20:fontsize=36:fontcolor=white:box=1:boxcolor=black@0.55[v4];
[5:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:black,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='Cam 5':x=20:y=20:fontsize=36:fontcolor=white:box=1:boxcolor=black@0.55[v5];
[v0][v1][v2][v3][v4][v5]xstack=inputs=6:layout=0_0|0_h0|w0_0|w0_h0|w0+w2_0|w0+w2_h0[v]
" \
  -map "[v]" \
  -c:v libx264 -crf 18 -preset medium \
  -shortest \
  combined_output_3x2.mp4
```

### 3D skeleton + camera grid side by side

```bash
ffmpeg -y \
  -i skeleton.mp4 \
  -i combined_output_3x2_5cams.mp4 \
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
