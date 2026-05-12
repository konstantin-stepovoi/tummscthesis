from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Callable, Optional
import traceback

import cv2
import numpy as np

from droplet_platform.acquisition import FrameSource, VideoRecorder, VideoRecorderConfig
from droplet_platform.detection import DetectorProtocol, detection_layer_from_frame
from .state import AppState


def make_display_frame(gray: np.ndarray, colorize: bool = True) -> np.ndarray:
    """Return a display-ready frame for OpenCV preview.

    Parameters
    ----------
    gray:
        Input mono frame. In normal camera/video operation this is a 2D uint8 image,
        but the function also tolerates already-color BGR frames.
    colorize:
        If True, apply an OpenCV false-color map to improve visual contrast during
        microscope focusing. If False, keep a neutral grayscale BGR image so colored
        overlays remain visible.

    Notes
    -----
    OpenCV overlays expect BGR frames. Therefore both branches return a 3-channel
    BGR image for 2D input.
    """
    if gray.ndim == 2:
        img = np.asarray(gray)
        if img.dtype != np.uint8:
            # Robust normalization for preview only; measurement uses the original frame.
            mn = float(np.nanmin(img)) if img.size else 0.0
            mx = float(np.nanmax(img)) if img.size else 1.0
            if mx > mn:
                img = ((img.astype(np.float32) - mn) * (255.0 / (mx - mn))).clip(0, 255).astype(np.uint8)
            else:
                img = np.zeros_like(img, dtype=np.uint8)

        if colorize:
            # TURBO gives high contrast while preserving monotonic intensity ordering.
            return cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Already color. Keep it unchanged; colorizing arbitrary BGR frames would destroy
    # the original colors and overlays from upstream sources.
    return gray.copy()


def draw_detections_on_frame(frame: np.ndarray, detection_layer: np.ndarray | None) -> np.ndarray:
    disp = frame.copy()
    arr = np.asarray(detection_layer) if detection_layer is not None else np.zeros((0, 4), dtype=np.float32)
    if arr.size == 0 or arr.ndim != 2 or arr.shape[1] < 3:
        return disp
    if disp.ndim == 2:
        disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
    for row in arr:
        y, x = int(round(row[1])), int(round(row[2]))
        cv2.drawMarker(disp, (x, y), (255, 180, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=1)
    return disp


def draw_tracks_on_frame(frame: np.ndarray, tracks_layer: np.ndarray | None, favorite_id: int | None = None, draw_ids: bool = True) -> np.ndarray:
    disp = frame.copy()
    if disp.ndim == 2:
        disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
    arr = np.asarray(tracks_layer) if tracks_layer is not None else np.zeros((0, 5), dtype=np.float32)
    if arr.size == 0 or arr.ndim != 2 or arr.shape[1] < 4:
        return disp
    for row in arr:
        tid = int(row[1])
        y, x = int(round(row[2])), int(round(row[3]))
        is_fav = favorite_id is not None and tid == int(favorite_id)
        color = (0, 255, 255) if is_fav else (0, 255, 0)
        cv2.drawMarker(disp, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=12, thickness=1)
        if draw_ids:
            cv2.putText(disp, str(tid), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return disp


def draw_finish_zone(frame: np.ndarray, state: AppState) -> np.ndarray:
    zone = state.favorite_zone
    if not zone.enabled:
        return frame
    disp = frame.copy()
    if disp.ndim == 2:
        disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
    cv2.line(disp, zone.p0, zone.p1, (255, 0, 255), max(1, int(zone.width_px // 20)))
    cv2.circle(disp, zone.p0, 4, (255, 0, 255), -1)
    cv2.circle(disp, zone.p1, 4, (255, 0, 255), -1)
    return disp


def check_favorite_crossing(state: AppState, tracks_layer: np.ndarray | None) -> bool:
    zone = state.favorite_zone
    if not zone.enabled or zone.favorite_id is None:
        return False
    arr = np.asarray(tracks_layer) if tracks_layer is not None else np.zeros((0, 5), dtype=np.float32)
    if arr.size == 0:
        return False
    for row in arr:
        if int(row[1]) == int(zone.favorite_id):
            return zone.contains_yx(float(row[2]), float(row[3]))
    return False


@dataclass
class PreviewConfig:
    window_name: str = "Droplet platform preview (q=stop)"
    colorize: bool = True
    draw_ids: bool = True
    draw_detections: bool = False
    beep_on_crossing: bool = True
    max_frames: int | None = None
    reset_tracker_on_start: bool = True
    record_raw_video_path: str | Path | None = None
    record_overlay_video_path: str | Path | None = None
    record_fps: float = 20.0
    video_codec: str = "MJPG"


class PreviewController:
    """Live/fake-stream preview loop with detection, tracking, overlays and optional video recording."""

    def __init__(self, source: FrameSource, detector: DetectorProtocol, state: AppState, cfg: PreviewConfig | None = None):
        self.source = source
        self.detector = detector
        self.state = state
        self.cfg = cfg or PreviewConfig()
        self.thread: Optional[threading.Thread] = None
        self.on_frame: Optional[Callable[[np.ndarray, np.ndarray], None]] = None
        self.on_error: Optional[Callable[[BaseException, str], None]] = None
        self.on_status: Optional[Callable[[str], None]] = None
        self._raw_recorder: VideoRecorder | None = None
        self._overlay_recorder: VideoRecorder | None = None
        self.last_exception: BaseException | None = None
        self.last_traceback: str | None = None

    def start_background(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.state.stop_event.clear()
        self.thread = threading.Thread(target=self.run, daemon=True, name="droplet-preview")
        self.thread.start()

    def stop(self) -> None:
        self.state.stop_event.set()

    def _status(self, msg: str) -> None:
        self.state.last_status = msg
        if self.on_status is not None:
            self.on_status(msg)

    def _init_recorders(self) -> None:
        self._raw_recorder = None
        self._overlay_recorder = None
        if self.cfg.record_raw_video_path:
            self._raw_recorder = VideoRecorder(VideoRecorderConfig(self.cfg.record_raw_video_path, fps=self.cfg.record_fps, codec=self.cfg.video_codec, is_color=False))
            self.state.raw_video_path = Path(self.cfg.record_raw_video_path)
        if self.cfg.record_overlay_video_path:
            self._overlay_recorder = VideoRecorder(VideoRecorderConfig(self.cfg.record_overlay_video_path, fps=self.cfg.record_fps, codec=self.cfg.video_codec, is_color=True))
            self.state.overlay_video_path = Path(self.cfg.record_overlay_video_path)

    def run(self) -> None:
        error_happened = False
        self.state.is_preview_running = True
        if self.cfg.reset_tracker_on_start:
            self.state.reset_stream_state(reset_tracker=True)
        self._init_recorders()
        frame_id = 0
        try:
            self._status("starting source")
            self.source.start()
            self._status("source started; opening preview window")
            cv2.namedWindow(self.cfg.window_name, cv2.WINDOW_NORMAL)
            while not self.state.stop_event.is_set():
                if self.cfg.max_frames is not None and frame_id >= self.cfg.max_frames:
                    break
                try:
                    gray, timestamp_ns = self.source.read()
                except EOFError:
                    self._status("end of video")
                    break

                det_layer = detection_layer_from_frame(gray, frame_id, self.detector)
                centers = det_layer[:, 1:3].astype(np.float32, copy=False) if det_layer.size else np.empty((0, 2), dtype=np.float32)
                scores = det_layer[:, 3].astype(np.float32, copy=False) if det_layer.size else np.empty((0,), dtype=np.float32)
                tracks = self.state.tracker.update(frame_id, centers, scores)

                disp = make_display_frame(gray, colorize=self.cfg.colorize)
                if self.cfg.draw_detections:
                    disp = draw_detections_on_frame(disp, det_layer)
                fav = self.state.favorite_zone.favorite_id if self.state.favorite_zone.enabled else None
                disp = draw_tracks_on_frame(disp, tracks, favorite_id=fav, draw_ids=self.cfg.draw_ids)
                disp = draw_finish_zone(disp, self.state)

                if check_favorite_crossing(self.state, tracks):
                    cv2.putText(disp, "COLLECT NOW", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    if self.cfg.beep_on_crossing:
                        try:
                            import winsound
                            winsound.Beep(1200, 150)
                        except Exception:
                            pass

                if self._raw_recorder is not None:
                    self._raw_recorder.write(gray)
                if self._overlay_recorder is not None:
                    self._overlay_recorder.write(disp)

                self.state.update_layers(gray, det_layer, tracks, timestamp_ns=timestamp_ns, display_frame=disp)
                if self.on_frame is not None:
                    self.on_frame(gray, tracks)

                cv2.imshow(self.cfg.window_name, disp)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
                frame_id += 1
        except BaseException as exc:
            error_happened = True
            self.last_exception = exc
            self.last_traceback = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            self.state.set_error(self.last_traceback)
            if self.on_error is not None:
                self.on_error(exc, self.last_traceback)
            else:
                print(self.last_traceback)
        finally:
            try:
                self.source.stop()
            finally:
                if self._raw_recorder is not None:
                    self._raw_recorder.close()
                if self._overlay_recorder is not None:
                    self._overlay_recorder.close()
                try:
                    cv2.destroyWindow(self.cfg.window_name)
                except Exception:
                    pass
                self.state.is_preview_running = False
                if error_happened:
                    self._status("preview stopped after error; see log")
                else:
                    self._status("preview stopped")
