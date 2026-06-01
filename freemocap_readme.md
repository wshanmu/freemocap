```bash
conda create -n freemocap python=3.11
conda activate freemocap
cd /data1/shanmu/ai-fitness-coach/freemocap
pip install -e .
```

```bash
python ./experimental/batch_process/headless_calibration.py
```

```bash
python ./freemocap/core_processes/process_motion_capture_videos/process_recording_headless.py
```
need to put the calibration file in the folder

``bash
python ./pose_recording/visualize_mediapipe_3d.py pose_recording/mediapipe_body_3d_xyz.csv --output ./pose_recording/test.mp4
```
consider using py39

for better visualization, to combined annotated videos:
```bash
ffmpeg \
  -i cam0_synced_mediapipe.mp4 \
  -i cam1_synced_mediapipe.mp4 \
  -i cam2_synced_mediapipe.mp4 \
  -i cam3_synced_mediapipe.mp4 \
  -i cam4_synced_mediapipe.mp4 \
  -i cam5_synced_mediapipe.mp4 \
  -filter_complex \
"[0:v][1:v][2:v][3:v][4:v][5:v]xstack=inputs=6:layout=0_0|0_h0|w0_0|w0_h0|w0+w2_0|w0+w2_h0[v]" \
  -map "[v]" \
  combined_output_3x2.mp4
```

```bash
ffmpeg -y \
  -i 3d_skeleton.mp4 \
  -i combined_output_3x2.mp4 \
  -filter_complex "[1:v]scale=-2:800[v1_scaled];[0:v][v1_scaled]hstack=inputs=2[v]" \
  -map "[v]" \
  -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p \
  output.mp4
```