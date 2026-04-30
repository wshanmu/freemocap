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
ffmpeg -i cam0_synced_mediapipe.mp4 -i cam1_synced_mediapipe.mp4 -i cam2_synced_mediapipe.mp4 -i cam3_synced_mediapipe.mp4 \
-filter_complex \ 
"[0:v][1:v][2:v][3:v]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]" \
-map "[v]" combined_output.mp4
```

```bash
ffmpeg -i test_0430.mp4 -i combined_output.mp4 -filter_complex "[1:v]scale=-1:800[v1_scaled]; [0:v][v1_scaled]hstack=inputs=2[v]" -map "[v]" output.mp4
```