from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Tuple
import time

import cv2
import numpy as np

try:  # optional: repository must import on non-Basler machines
    from pypylon import pylon
except Exception:  # pragma: no cover
    pylon = None


class FrameSource(Protocol):
    """Common frame-source interface used by preview and recording code."""

    def start(self) -> None: ...
    def read(self) -> Tuple[np.ndarray, int]: ...
    def stop(self) -> None: ...


@dataclass
class CameraConfig:
    fps: float = 20.0
    exposure_ms: float = 50.0
    timeout_ms: int = 5000
    mono8: bool = True
    device_index: int = 0
    latest_only: bool = True


def _set_first_available_feature(node_map_obj, names: tuple[str, ...], value) -> bool:
    """Set the first camera feature that exists and is writable.

    Basler feature names differ across models and pylon versions. The helper is
    intentionally permissive so the same code can run on several cameras.
    """
    for name in names:
        try:
            node = getattr(node_map_obj, name)
            node.SetValue(value)
            return True
        except Exception:
            continue
    return False


class BaslerCameraSource:
    """Basler/pypylon source returning `(mono8_frame, timestamp_ns)`.

    This class replaces the camera-access code that previously lived inside the
    acquisition notebook. It owns configuration, start/stop, and conversion to
    mono8 grayscale frames.
    """

    def __init__(self, cfg: CameraConfig | None = None):
        if pylon is None:
            raise RuntimeError(
                "pypylon is not available. Install Basler pylon + pypylon on the acquisition PC."
            )
        self.cfg = cfg or CameraConfig()
        self.cam = None
        self.converter = None

    def start(self) -> None:
        if self.cam is not None:
            return
        factory = pylon.TlFactory.GetInstance()
        devices = factory.EnumerateDevices()
        if len(devices) == 0:
            raise RuntimeError("No Basler camera detected")
        idx = min(max(int(self.cfg.device_index), 0), len(devices) - 1)
        self.cam = pylon.InstantCamera(factory.CreateDevice(devices[idx]))
        self.cam.Open()
        self.configure(self.cfg)

        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_Mono8
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        strategy = pylon.GrabStrategy_LatestImageOnly if self.cfg.latest_only else pylon.GrabStrategy_OneByOne
        self.cam.StartGrabbing(strategy)

    def configure(self, cfg: CameraConfig | None = None) -> None:
        """Apply camera parameters. Safe to call during preview."""
        if cfg is not None:
            self.cfg = cfg
        if self.cam is None:
            return
        cam = self.cam
        exposure_us = float(self.cfg.exposure_ms) * 1000.0
        _set_first_available_feature(cam, ("ExposureTime", "ExposureTimeAbs"), exposure_us)
        try:
            cam.AcquisitionFrameRateEnable.SetValue(True)
        except Exception:
            pass
        _set_first_available_feature(cam, ("AcquisitionFrameRate", "AcquisitionFrameRateAbs"), float(self.cfg.fps))
        if self.cfg.mono8:
            try:
                cam.PixelFormat.SetValue("Mono8")
            except Exception:
                pass

    def read(self) -> Tuple[np.ndarray, int]:
        if self.cam is None or self.converter is None:
            raise RuntimeError("Camera source is not started")
        grab = self.cam.RetrieveResult(self.cfg.timeout_ms, pylon.TimeoutHandling_ThrowException)
        try:
            if not grab.GrabSucceeded():
                raise RuntimeError("Basler frame grab failed")
            gray = self.converter.Convert(grab).GetArray()
            timestamp_ns = int(getattr(grab, "TimeStamp", time.time_ns()))
            return gray, timestamp_ns
        finally:
            grab.Release()

    def stop(self) -> None:
        try:
            if self.cam is not None:
                try:
                    if self.cam.IsGrabbing():
                        self.cam.StopGrabbing()
                finally:
                    self.cam.Close()
        finally:
            self.cam = None
            self.converter = None

    def snapshot(self) -> Tuple[np.ndarray, int]:
        """Start if needed, grab one frame, then stop only if started here."""
        started_here = self.cam is None
        if started_here:
            self.start()
        try:
            return self.read()
        finally:
            if started_here:
                self.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False


class VideoFileSource:
    """OpenCV video-file source with the same interface as `BaslerCameraSource`.

    It is used both for analysis and for fake streaming during demos or detector
    parameter tuning without access to the live camera.
    """

    def __init__(self, path: str | Path, loop: bool = False, grayscale: bool = True, realtime_fps: float | None = None):
        self.path = Path(path).expanduser()
        self.loop = bool(loop)
        self.grayscale = bool(grayscale)
        self.realtime_fps = None if realtime_fps in (None, 0) else float(realtime_fps)
        self.cap: Optional[cv2.VideoCapture] = None
        self._last_read_t: float | None = None

    def start(self) -> None:
        if self.cap is not None:
            return
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {self.path}")
        self._last_read_t = None

    def read(self) -> Tuple[np.ndarray, int]:
        if self.cap is None:
            raise RuntimeError("Video source is not started")
        if self.realtime_fps:
            now = time.perf_counter()
            if self._last_read_t is not None:
                wait = (1.0 / self.realtime_fps) - (now - self._last_read_t)
                if wait > 0:
                    time.sleep(wait)
            self._last_read_t = time.perf_counter()
        ret, frame = self.cap.read()
        if not ret:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            if not ret:
                raise EOFError("End of video")
        if self.grayscale and frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        timestamp_ns = time.time_ns()
        return frame, timestamp_ns

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
        self.cap = None

    def frame_count(self) -> int:
        if self.cap is None:
            cap = cv2.VideoCapture(str(self.path))
            try:
                return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            finally:
                cap.release()
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def fps(self) -> float:
        if self.cap is None:
            cap = cv2.VideoCapture(str(self.path))
            try:
                return float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            finally:
                cap.release()
        return float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False


def make_camera_source(cfg: CameraConfig | None = None) -> BaslerCameraSource:
    return BaslerCameraSource(cfg or CameraConfig())


def make_video_source(path: str | Path, loop: bool = False, realtime_fps: float | None = None) -> VideoFileSource:
    return VideoFileSource(path, loop=loop, realtime_fps=realtime_fps)
