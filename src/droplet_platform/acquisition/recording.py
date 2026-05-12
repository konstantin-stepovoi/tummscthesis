from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .camera import FrameSource


@dataclass
class VideoRecorderConfig:
    output_path: str | Path
    fps: float = 20.0
    codec: str = "MJPG"
    is_color: bool = False


class VideoRecorder:
    """Small OpenCV VideoWriter wrapper for raw or overlay video recording."""

    def __init__(self, cfg: VideoRecorderConfig):
        self.cfg = cfg
        self.writer: Optional[cv2.VideoWriter] = None
        self.output_path = Path(cfg.output_path).expanduser()

    def start(self, first_frame: np.ndarray) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        h, w = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*str(self.cfg.codec)[:4])
        is_color = bool(self.cfg.is_color or (first_frame.ndim == 3))
        self.writer = cv2.VideoWriter(str(self.output_path), fourcc, float(self.cfg.fps), (w, h), is_color)
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot open VideoWriter for {self.output_path}")

    def write(self, frame: np.ndarray) -> None:
        if self.writer is None:
            self.start(frame)
        assert self.writer is not None
        if frame.ndim == 2 and self.cfg.is_color:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.ndim == 3 and not self.cfg.is_color:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.writer.write(frame)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
        self.writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def record_raw_video(source: FrameSource, output_path: str | Path, n_frames: int, fps: float, codec: str = "MJPG") -> Path:
    """Record raw grayscale frames from a source to a video file."""
    out = Path(output_path).expanduser()
    source.start()
    recorder = VideoRecorder(VideoRecorderConfig(out, fps=fps, codec=codec, is_color=False))
    try:
        for _ in range(int(n_frames)):
            gray, _ = source.read()
            recorder.write(gray)
    finally:
        recorder.close()
        source.stop()
    return out
