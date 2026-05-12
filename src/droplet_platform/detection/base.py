from __future__ import annotations

from typing import Protocol, Tuple, Union, runtime_checkable
import numpy as np

DetectionReturn = Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]


@runtime_checkable
class DetectorProtocol(Protocol):
    """Detector interface shared by preview, tracker, and recorder.

    `detect_centers()` returns droplet centers in detector-grid coordinates. The
    conversion to full-resolution coordinates is centralized in
    `detection_layer_from_frame()` through the detector's `ds` attribute.
    """

    ds: int

    def detect_centers(
        self,
        frame_gray: np.ndarray,
        exclude_border: bool = False,
        return_scores: bool = False,
    ) -> DetectionReturn:
        ...


def detection_layer_from_frame(frame_gray: np.ndarray, frame_idx: int, detector: DetectorProtocol) -> np.ndarray:
    """Return compact full-resolution detection layer: [frame_id, y, x, score]."""
    centers_ds, scores = detector.detect_centers(frame_gray, exclude_border=False, return_scores=True)
    centers_ds = np.asarray(centers_ds)
    scores = np.asarray(scores, dtype=np.float32)
    if centers_ds.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    centers_ds = centers_ds.reshape(-1, 2).astype(np.float32, copy=False)
    ds = int(getattr(detector, "ds", 1))
    centers_full = centers_ds.copy()
    centers_full[:, 0] *= ds
    centers_full[:, 1] *= ds
    out = np.empty((centers_full.shape[0], 4), dtype=np.float32)
    out[:, 0] = float(frame_idx)
    out[:, 1:3] = centers_full
    if scores.size != centers_full.shape[0]:
        scores = np.zeros((centers_full.shape[0],), dtype=np.float32)
    out[:, 3] = scores.astype(np.float32, copy=False)
    return out
