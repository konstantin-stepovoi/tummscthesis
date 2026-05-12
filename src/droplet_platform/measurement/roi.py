from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping
import numpy as np


@dataclass(frozen=True)
class DiskROI:
    radius_px: int
    offsets: np.ndarray

    @classmethod
    def make(cls, radius_px: int) -> "DiskROI":
        return cls(radius_px=int(radius_px), offsets=precompute_disk_offsets(radius_px))


def precompute_disk_offsets(radius: int) -> np.ndarray:
    """Return `(dy, dx)` offsets for all pixels inside a disk."""
    r = int(radius)
    if r < 0:
        raise ValueError("radius must be non-negative")
    ys, xs = np.ogrid[-r : r + 1, -r : r + 1]
    mask = (ys * ys + xs * xs) <= (r * r)
    dy, dx = np.where(mask)
    dy = dy - r
    dx = dx - r
    return np.stack([dy, dx], axis=1).astype(np.int16)


def sum_in_disk(img_u8: np.ndarray, cy: float, cx: float, offsets: np.ndarray) -> float:
    """Sum pixel intensities inside a disk centered at `(cy, cx)`.

    Returns NaN if the disk has no valid pixels in the image.
    """
    if img_u8.ndim != 2:
        raise ValueError("sum_in_disk expects a grayscale 2D image")
    h, w = img_u8.shape
    y0 = int(round(float(cy)))
    x0 = int(round(float(cx)))
    yy = y0 + offsets[:, 0].astype(np.int32)
    xx = x0 + offsets[:, 1].astype(np.int32)
    valid = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
    if not np.any(valid):
        return float("nan")
    return float(img_u8[yy[valid], xx[valid]].sum(dtype=np.int64))


def positions_from_track_layer(track_layer: np.ndarray) -> dict[int, tuple[float, float]]:
    """Convert `[frame,id,y,x,score]` tracker layer to `{id: (y, x)}` mapping."""
    arr = np.asarray(track_layer) if track_layer is not None else np.zeros((0, 5), dtype=np.float32)
    if arr.size == 0 or arr.ndim != 2 or arr.shape[1] < 4:
        return {}
    out: dict[int, tuple[float, float]] = {}
    for row in arr:
        tid = int(row[1])
        if tid <= 0:
            continue
        out[tid] = (float(row[2]), float(row[3]))
    return out


def measure_ids_in_frame(
    img_u8: np.ndarray,
    track_layer: np.ndarray,
    ids: Iterable[int],
    offsets: np.ndarray,
) -> dict[int, float]:
    """Measure integrated intensity for selected track IDs in one frame."""
    positions = positions_from_track_layer(track_layer)
    result: dict[int, float] = {}
    for rid in ids:
        rid = int(rid)
        if rid not in positions:
            result[rid] = float("nan")
            continue
        y, x = positions[rid]
        result[rid] = sum_in_disk(img_u8, y, x, offsets)
    return result
