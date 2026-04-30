"""
Visualize camera positions and orientations from a FreeMoCap calibration TOML.
Saves an interactive HTML (plotly) and a static PNG (matplotlib).

Usage:
    python visualize_calibration.py [path/to/calibration.toml]
"""

import sys
from pathlib import Path

import toml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import plotly.graph_objects as go


def load_cameras(toml_path: Path):
    data = toml.load(toml_path)
    cameras = {}
    for key, val in data.items():
        if key.startswith("cam_"):
            cameras[key] = {
                "name": val["name"],
                "position": np.array(val["world_position"]),
                "orientation": np.array(val["world_orientation"]),  # 3x3, rows = cam X/Y/Z in world
            }
    return cameras


def draw_camera_axes_matplotlib(ax, position, orientation, scale, cam_name):
    """Draw RGB axes (X=red, Y=green, Z=blue) for one camera."""
    colors = ["red", "green", "blue"]
    labels = ["X", "Y", "Z"]
    for i, (color, label) in enumerate(zip(colors, labels)):
        direction = orientation[i]  # i-th row = camera's i-th axis in world coords
        ax.quiver(
            *position,
            *(direction * scale),
            color=color,
            linewidth=1.5,
            arrow_length_ratio=0.2,
        )
    ax.text(*position, f"  {cam_name}", fontsize=8, color="black")


def visualize_matplotlib(cameras: dict, out_path: Path, axis_scale: float):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    positions = np.array([c["position"] for c in cameras.values()])

    for cam_id, cam in cameras.items():
        draw_camera_axes_matplotlib(ax, cam["position"], cam["orientation"], axis_scale, cam["name"])

    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=60, color="black", zorder=5)

    # World origin
    ax.scatter([0], [0], [0], s=100, color="purple", marker="*", label="Origin (cam_0)", zorder=6)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Camera Calibration — World Positions & Orientations\n(R=X, G=Y, B=Z axes)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved PNG → {out_path}")
    plt.close()


def visualize_plotly(cameras: dict, out_path: Path, axis_scale: float):
    traces = []

    positions = np.array([c["position"] for c in cameras.values()])
    names = [c["name"] for c in cameras.values()]

    # Camera position markers
    traces.append(go.Scatter3d(
        x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
        mode="markers+text",
        text=names,
        textposition="top center",
        marker=dict(size=6, color="black"),
        name="Cameras",
    ))

    # World origin
    traces.append(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers+text",
        text=["Origin"],
        textposition="top center",
        marker=dict(size=8, color="purple", symbol="diamond"),
        name="Origin",
    ))

    axis_colors = ["red", "green", "blue"]
    axis_labels = ["X", "Y", "Z"]

    for cam_id, cam in cameras.items():
        pos = cam["position"]
        ori = cam["orientation"]
        for i, (color, label) in enumerate(zip(axis_colors, axis_labels)):
            tip = pos + ori[i] * axis_scale
            traces.append(go.Scatter3d(
                x=[pos[0], tip[0]],
                y=[pos[1], tip[1]],
                z=[pos[2], tip[2]],
                mode="lines",
                line=dict(color=color, width=4),
                name=f"{cam['name']} {label}",
                showlegend=(cam_id == "cam_0"),  # only show legend once per axis color
                legendgroup=f"axis_{label}",
            ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Camera Calibration — World Positions & Orientations",
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",
        ),
        legend=dict(title="Legend"),
    )

    fig.write_html(str(out_path))
    print(f"Saved HTML → {out_path}")


if __name__ == "__main__":
    toml_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "videos/calib_videos/videos_camera_calibration.toml"
    )

    if not toml_path.exists():
        print(f"TOML not found: {toml_path}")
        sys.exit(1)

    cameras = load_cameras(toml_path)
    print(f"Loaded {len(cameras)} cameras from {toml_path}")
    for cam_id, cam in cameras.items():
        print(f"  {cam_id}: position={cam['position'].round(1)}")

    out_dir = toml_path.parent

    # Scale axis arrows to ~15% of the max spread between cameras
    positions = np.array([c["position"] for c in cameras.values()])
    spread = np.linalg.norm(positions.max(axis=0) - positions.min(axis=0))
    axis_scale = spread * 0.15

    visualize_matplotlib(cameras, out_dir / "camera_calibration.png", axis_scale)
    visualize_plotly(cameras, out_dir / "camera_calibration.html", axis_scale)
