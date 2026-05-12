from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import threading
import time
import traceback

import numpy as np

from droplet_platform.tracking import TrackerService


@dataclass
class FavoriteZone:
    enabled: bool = False
    favorite_id: Optional[int] = None
    p0: tuple[int, int] = (0, 0)  # (x,y)
    p1: tuple[int, int] = (300, 0)
    width_px: int = 80

    def contains_yx(self, y: float, x: float) -> bool:
        ax, ay = self.p0
        bx, by = self.p1
        px, py = float(x), float(y)
        vx, vy = bx - ax, by - ay
        wx, wy = px - ax, py - ay
        denom = vx * vx + vy * vy
        if denom <= 1e-9:
            return False
        t = (wx * vx + wy * vy) / denom
        if t < 0.0 or t > 1.0:
            return False
        projx, projy = ax + t * vx, ay + t * vy
        dist2 = (px - projx) ** 2 + (py - projy) ** 2
        return dist2 <= (float(self.width_px) / 2.0) ** 2


@dataclass
class AppState:
    root_dir: Path = Path("experiments")
    tracker: TrackerService = field(default_factory=TrackerService)
    last_frame: Optional[np.ndarray] = None
    last_display_frame: Optional[np.ndarray] = None
    last_detection_layer: Optional[np.ndarray] = None
    last_tracks_layer: Optional[np.ndarray] = None
    last_timestamp_ns: Optional[int] = None
    frame_id: int = 0
    is_preview_running: bool = False
    is_recording: bool = False
    favorite_zone: FavoriteZone = field(default_factory=FavoriteZone)
    stop_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_status: str = "idle"
    last_error: Optional[str] = None
    raw_video_path: Optional[Path] = None
    overlay_video_path: Optional[Path] = None
    last_source_kind: str = "camera"
    last_update_time: float = field(default_factory=time.time)

    def reset_stream_state(self, reset_tracker: bool = True) -> None:
        with self.lock:
            self.last_frame = None
            self.last_display_frame = None
            self.last_detection_layer = None
            self.last_tracks_layer = None
            self.last_timestamp_ns = None
            self.frame_id = 0
            self.last_error = None
            if reset_tracker:
                self.tracker.reset()
            self.stop_event.clear()
            self.last_update_time = time.time()

    def active_ids(self) -> list[int]:
        return self.tracker.ids()

    def update_layers(
        self,
        frame: np.ndarray,
        detection_layer: np.ndarray,
        tracks_layer: np.ndarray,
        *,
        timestamp_ns: int | None = None,
        display_frame: np.ndarray | None = None,
    ) -> None:
        with self.lock:
            self.last_frame = frame.copy() if frame is not None else None
            self.last_display_frame = display_frame.copy() if display_frame is not None else None
            self.last_detection_layer = detection_layer.copy() if detection_layer is not None else None
            self.last_tracks_layer = tracks_layer.copy() if tracks_layer is not None else None
            self.last_timestamp_ns = timestamp_ns
            self.frame_id += 1
            self.last_update_time = time.time()

    def set_error(self, exc: BaseException | str) -> str:
        if isinstance(exc, BaseException):
            msg = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        else:
            msg = str(exc)
        with self.lock:
            self.last_error = msg
            self.last_status = "error"
        return msg

    def save_last_frame(self, path: str | Path, *, display: bool = False) -> Path:
        import cv2

        out = Path(path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            frame = self.last_display_frame if display else self.last_frame
            if frame is None:
                raise RuntimeError("No frame has been acquired yet")
            frame_to_save = frame.copy()
        if not cv2.imwrite(str(out), frame_to_save):
            raise RuntimeError(f"Could not save frame to {out}")
        return out
