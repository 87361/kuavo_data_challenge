#!/usr/bin/env python3
# Usage:
#   python scripts/gui/lerobot_editor/trajectory_video.py
#   python scripts/gui/lerobot_editor/trajectory_video.py --episode 0 --max-frames 120 --output /tmp/episode_0_preview.mp4
#   python scripts/gui/lerobot_editor/trajectory_video.py --episode 12 --hand right --source action --output /tmp/episode_12_action.mp4
#   python scripts/gui/lerobot_editor/trajectory_video.py --dataset /path/to/lerobot --urdf /path/to/biped_s45.urdf --video-key observation.images.head_cam_h
#
# The output video shows a 3D end-effector trajectory that grows in recorded
# time order next to the synchronized head-camera frames from the same episode.
# It uses the dataset FPS by default, so playback time matches the real record.

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterator

import av
import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")

try:
    from .dataset import LeRobotV21Dataset, load_v21_dataset
    from .trajectory_visualizer import (
        DEFAULT_DATASET,
        DEFAULT_URDF,
        load_episode_positions,
        path_length,
        selected_hands,
        set_axes_equal,
    )
    from .urdf_fk import SimpleArmFk
except ImportError:  # pragma: no cover - direct script execution fallback
    from dataset import LeRobotV21Dataset, load_v21_dataset
    from trajectory_visualizer import (
        DEFAULT_DATASET,
        DEFAULT_URDF,
        load_episode_positions,
        path_length,
        selected_hands,
        set_axes_equal,
    )
    from urdf_fk import SimpleArmFk


DEFAULT_OUTPUT = Path("/tmp/kuavo_eef_trajectory_episode_000000_with_head_cam.mp4")
DEFAULT_VIDEO_KEY = "observation.images.head_cam_h"
ProgressCallback = Callable[[dict[str, Any]], None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a time-synchronized Kuavo EEF trajectory + camera video.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="LeRobot v2.1 dataset root")
    parser.add_argument("--urdf", default=str(DEFAULT_URDF), help="Kuavo URDF used for FK")
    parser.add_argument("--episode", type=int, default=0, help="episode index to render")
    parser.add_argument("--video-key", default=DEFAULT_VIDEO_KEY, help="video feature shown beside the trajectory")
    parser.add_argument(
        "--source",
        choices=["state", "action"],
        default="state",
        help="column to visualize: observation.state or action",
    )
    parser.add_argument(
        "--hand",
        choices=["auto", "left", "right", "both"],
        default="auto",
        help="which end-effector trajectory to plot; auto selects the longer path",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="output MP4 path")
    parser.add_argument("--title", default=None, help="trajectory panel title")
    parser.add_argument("--width", type=int, default=1280, help="output video width")
    parser.add_argument("--height", type=int, default=720, help="output video height")
    parser.add_argument("--dpi", type=int, default=120, help="matplotlib panel DPI")
    parser.add_argument("--stride", type=int, default=1, help="render every Nth recorded frame")
    parser.add_argument("--fps", type=float, default=None, help="output FPS; defaults to dataset_fps / stride")
    parser.add_argument("--max-frames", type=int, default=None, help="limit output frames for quick previews")
    parser.add_argument("--camera-left", action="store_true", help="place camera on the left and trajectory on the right")
    parser.add_argument("--codec", default="h264", help="output codec: h264 for browser MP4, or an OpenCV fourcc such as mp4v")
    return parser.parse_args()


def iter_video_rgb(video_path: Path) -> Iterator[np.ndarray]:
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            yield frame.to_ndarray(format="rgb24")


def next_frame_for_index(
    decoder: Iterator[np.ndarray],
    target_index: int,
    decoded_index: int,
    last_frame: np.ndarray | None,
) -> tuple[np.ndarray, int, np.ndarray]:
    while decoded_index <= target_index:
        try:
            last_frame = next(decoder)
        except StopIteration:
            break
        decoded_index += 1
    if last_frame is None:
        raise RuntimeError("video stream produced no frames")
    return last_frame, decoded_index, last_frame


def letterbox_rgb(image: np.ndarray, width: int, height: int, fill: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    panel = np.full((height, width, 3), fill, dtype=np.uint8)
    src_h, src_w = image.shape[:2]
    scale = min(width / src_w, height / src_h)
    dst_w = max(1, int(round(src_w * scale)))
    dst_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(image, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
    x0 = (width - dst_w) // 2
    y0 = (height - dst_h) // 2
    panel[y0 : y0 + dst_h, x0 : x0 + dst_w] = resized
    return panel


def annotate_camera_panel(panel: np.ndarray, label: str, frame_index: int, timestamp: float) -> None:
    text = f"{label}  frame {frame_index}  t={timestamp:.2f}s"
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 48), (0, 0, 0), thickness=-1)
    cv2.putText(panel, text, (18, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2, cv2.LINE_AA)


class TrajectoryPanel:
    def __init__(
        self,
        selected: list[tuple[str, np.ndarray, str]],
        title: str,
        width: int,
        height: int,
        dpi: int,
        source_fps: float,
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.lines import Line2D

        self.selected = selected
        self.source_fps = source_fps
        self.dynamic_artists: list[matplotlib.artist.Artist] = []

        self.fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="black")
        self.canvas = FigureCanvasAgg(self.fig)
        self.ax = self.fig.add_subplot(111, projection="3d", facecolor="white")
        self.fig.subplots_adjust(left=0.06, right=0.91, top=0.96, bottom=0.21)
        title_size = 13 if len(title) <= 54 else 11
        self.title_text = self.fig.text(
            0.5,
            0.07,
            title,
            color="white",
            fontsize=title_size,
            fontweight="bold",
            ha="center",
            va="center",
        )
        self.time_text = self.fig.text(0.5, 0.032, "", color="#d9dde2", fontsize=9, ha="center", va="center")

        set_axes_equal(self.ax, [points for _, points, _ in selected])
        self.ax.view_init(elev=23, azim=-50)
        try:
            self.ax.set_box_aspect((1, 1, 0.82))
        except Exception:
            pass
        self.ax.set_xlabel(r"$x_e(m)$", labelpad=7)
        self.ax.set_ylabel(r"$y_e(m)$", labelpad=7)
        self.ax.set_zlabel(r"$z_e(m)$", labelpad=7)
        self.ax.grid(True, alpha=0.42)
        self.ax.tick_params(axis="both", which="major", labelsize=7)
        self.ax.tick_params(axis="z", which="major", pad=-2, labelsize=7)
        for tick_label in [
            *self.ax.xaxis.get_ticklabels(),
            *self.ax.yaxis.get_ticklabels(),
            *self.ax.zaxis.get_ticklabels(),
        ]:
            tick_label.set_clip_on(False)

        handles = [
            Line2D([0], [0], color="black", lw=2.4, label=f"{label} path")
            for label, _, _ in selected
        ]
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#d62828",
                markeredgecolor="black",
                markersize=7,
                label="current",
            )
        )
        self.ax.legend(handles=handles, loc="upper left", fontsize=7, frameon=True)

        for label, points, cmap_name in selected:
            start_color = matplotlib.colormaps[cmap_name](0.05)
            self.ax.scatter(*points[0], s=32, color=start_color, edgecolor="black", linewidth=0.4)

    def render(self, frame_index: int, source_frame_index: int) -> np.ndarray:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        for artist in self.dynamic_artists:
            artist.remove()
        self.dynamic_artists.clear()

        denominator = max(max(len(points) for _, points, _ in self.selected) - 1, 1)
        for _, points, cmap_name in self.selected:
            upto = min(frame_index + 1, len(points))
            visible = points[:upto]
            if len(visible) >= 2:
                segments = np.stack([visible[:-1], visible[1:]], axis=1)
                collection = Line3DCollection(segments, cmap=cmap_name, linewidth=2.3, alpha=0.97)
                collection.set_array(np.linspace(0.0, (upto - 1) / denominator, len(segments)))
                collection.set_clim(0.0, 1.0)
                self.ax.add_collection3d(collection)
                self.dynamic_artists.append(collection)
            current = self.ax.scatter(
                *visible[-1],
                s=46,
                marker="o",
                color="#d62828",
                edgecolor="black",
                linewidth=0.45,
                depthshade=True,
            )
            self.dynamic_artists.append(current)

        timestamp = source_frame_index / self.source_fps
        self.time_text.set_text(f"frame {source_frame_index}   t={timestamp:.2f}s")
        self.canvas.draw()
        width, height = self.canvas.get_width_height()
        rgba = np.frombuffer(self.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
        return rgba[:, :, :3].copy()


def output_indices(length: int, stride: int, max_frames: int | None) -> list[int]:
    indices = list(range(0, length, stride))
    if max_frames is not None:
        indices = indices[: max(0, max_frames)]
    if not indices:
        raise ValueError("no frames selected; adjust --stride or --max-frames")
    return indices


def build_title(args: argparse.Namespace) -> str:
    if args.title:
        return args.title
    source_label = "observation.state" if args.source == "state" else "action"
    return f"Kuavo EEF Trajectory · episode {args.episode} · {source_label}"


def open_writer(output_path: Path, codec: str, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*codec[:4])
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {output_path} with codec {codec!r}")
    return writer


def wants_h264(codec: str) -> bool:
    return codec.lower() in {"h264", "avc1", "libx264"}


def transcode_to_h264(input_path: Path, output_path: Path, fps: float) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-r",
        f"{fps:.8g}",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to write browser-compatible H.264 previews") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"ffmpeg H.264 transcode failed: {detail}") from exc


def render_video(
    dataset: LeRobotV21Dataset,
    args: argparse.Namespace,
    left: np.ndarray,
    right: np.ndarray,
    progress: ProgressCallback | None = None,
) -> Path:
    video_path = dataset.video_path(args.episode, args.video_key)
    if not video_path.exists():
        raise FileNotFoundError(f"camera video not found: {video_path}")

    selected = selected_hands(args.hand, left, right)
    base_fps = float(dataset.fps)
    stride = max(1, int(args.stride))
    fps = float(args.fps) if args.fps else base_fps / stride
    indices = output_indices(len(left), stride, args.max_frames)
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    h264_output = wants_h264(str(args.codec))
    write_path = output_path.with_name(f"{output_path.stem}.tmp_mpeg4{output_path.suffix}") if h264_output else output_path
    if write_path.exists():
        write_path.unlink()

    width = max(640, int(args.width))
    height = max(360, int(args.height))
    camera_width = width // 2
    plot_width = width - camera_width
    if args.camera_left:
        left_panel_name, right_panel_name = "camera", "trajectory"
    else:
        left_panel_name, right_panel_name = "trajectory", "camera"

    panel = TrajectoryPanel(
        selected=selected,
        title=build_title(args),
        width=plot_width,
        height=height,
        dpi=max(72, int(args.dpi)),
        source_fps=base_fps,
    )
    video_frame_offset = dataset.video_frame_offset(args.episode, args.video_key)
    decoder = iter_video_rgb(video_path)
    decoded_index = 0
    last_camera_frame: np.ndarray | None = None
    writer = open_writer(write_path, "mp4v" if h264_output else args.codec, fps, (width, height))

    try:
        for out_idx, source_idx in enumerate(indices):
            camera_rgb, decoded_index, last_camera_frame = next_frame_for_index(
                decoder,
                video_frame_offset + source_idx,
                decoded_index,
                last_camera_frame,
            )
            plot_rgb = panel.render(out_idx, source_idx)
            plot_rgb = cv2.resize(plot_rgb, (plot_width, height), interpolation=cv2.INTER_AREA)
            camera_panel = letterbox_rgb(camera_rgb, camera_width, height)
            annotate_camera_panel(camera_panel, args.video_key, source_idx, source_idx / base_fps)

            panels = {
                "trajectory": plot_rgb,
                "camera": camera_panel,
            }
            combined_rgb = np.concatenate([panels[left_panel_name], panels[right_panel_name]], axis=1)
            writer.write(cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR))

            if out_idx == 0 or (out_idx + 1) % 50 == 0 or out_idx + 1 == len(indices):
                message = f"Rendered {out_idx + 1}/{len(indices)} frames"
                print(message)
                if progress is not None:
                    progress(
                        {
                            "status": "running",
                            "message": message,
                            "progress": 0.25 + 0.7 * ((out_idx + 1) / max(1, len(indices))),
                        }
                    )
    finally:
        writer.release()

    if h264_output:
        if progress is not None:
            progress({"status": "running", "message": "encoding browser-compatible MP4", "progress": 0.97})
        transcode_to_h264(write_path, output_path, fps)
        write_path.unlink(missing_ok=True)

    return output_path


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be >= 1")

    dataset_path = Path(args.dataset).expanduser().resolve()
    urdf_path = Path(args.urdf).expanduser().resolve()
    dataset = load_v21_dataset(dataset_path)
    fk = SimpleArmFk(urdf_path)
    left, right = load_episode_positions(dataset, args.episode, fk, args.source)

    print(
        "Episode",
        args.episode,
        f"frames={len(left)}",
        f"fps={float(dataset.fps):.3g}",
        f"left_length={path_length(left):.4f}m",
        f"right_length={path_length(right):.4f}m",
    )
    output_path = render_video(dataset, args, left, right)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
