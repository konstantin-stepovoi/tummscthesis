from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional
import csv
import time

import numpy as np

from droplet_platform.acquisition import FrameSource
from droplet_platform.detection import DetectorProtocol, detection_layer_from_frame
from droplet_platform.tracking import TrackerService
from .roi import precompute_disk_offsets, measure_ids_in_frame

ProgressCallback = Callable[[int, int, np.ndarray | None], None]


@dataclass
class RecorderConfig:
    output_path: str | Path
    duration_s: float
    fps: float
    radius_px: int
    ids_to_track: list[int] = field(default_factory=list)
    write_timestamp: bool = True
    flush_every: int = 25


class DropletIntensityRecorder:
    """Record per-droplet integrated intensity into a TSV table.

    This is the cleaned-up version of the old `Grid_measurer.py` logic. It keeps
    detection, tracking, and droplet-wise photometry synchronized frame-by-frame:

    frame -> detector -> tracker -> ROI sums -> TSV row.
    """

    def __init__(
        self,
        source: FrameSource,
        detector: DetectorProtocol,
        tracker: TrackerService,
        cfg: RecorderConfig,
    ):
        self.source = source
        self.detector = detector
        self.tracker = tracker
        self.cfg = cfg
        self.offsets = precompute_disk_offsets(cfg.radius_px)
        self.stop_requested = False

    def stop(self) -> None:
        self.stop_requested = True

    def _resolve_ids(self) -> list[int]:
        ids = [int(x) for x in self.cfg.ids_to_track if int(x) > 0]
        if ids:
            return sorted(set(ids))
        ids = self.tracker.ids()
        if not ids:
            raise RuntimeError("No droplet IDs selected and tracker has no active IDs")
        return ids

    def run(self, progress: ProgressCallback | None = None) -> Path:
        out = Path(self.cfg.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        ids = self._resolve_ids()
        n_frames = int(round(float(self.cfg.duration_s) * float(self.cfg.fps)))
        n_frames = max(1, n_frames)

        started_here = False
        try:
            self.source.start()
            started_here = True
        except Exception:
            # Source may already be started by preview; allow caller-owned source.
            started_here = False

        try:
            with out.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh, delimiter="\t")
                header = ["timestamp_ns"] + [f"id{rid:04d}" for rid in ids]
                writer.writerow(header)

                for frame_id in range(n_frames):
                    if self.stop_requested:
                        break
                    gray, timestamp_ns = self.source.read()
                    layer = detection_layer_from_frame(gray, frame_id, self.detector)
                    centers = layer[:, 1:3].astype(np.float32, copy=False) if layer.size else np.empty((0, 2), dtype=np.float32)
                    scores = layer[:, 3].astype(np.float32, copy=False) if layer.size else np.empty((0,), dtype=np.float32)
                    tracks = self.tracker.update(frame_id, centers, scores)
                    values = measure_ids_in_frame(gray, tracks, ids, self.offsets)
                    row = [str(int(timestamp_ns))] + ["nan" if np.isnan(values[rid]) else f"{values[rid]:.0f}" for rid in ids]
                    writer.writerow(row)
                    if self.cfg.flush_every and frame_id % int(self.cfg.flush_every) == 0:
                        fh.flush()
                    if progress is not None:
                        progress(frame_id + 1, n_frames, tracks)
        finally:
            if started_here:
                self.source.stop()
        return out


def record_droplet_intensities(
    source: FrameSource,
    detector: DetectorProtocol,
    tracker: TrackerService,
    output_path: str | Path,
    duration_s: float,
    fps: float,
    radius_px: int,
    ids_to_track: Optional[Iterable[int]] = None,
    progress: ProgressCallback | None = None,
) -> Path:
    cfg = RecorderConfig(
        output_path=output_path,
        duration_s=duration_s,
        fps=fps,
        radius_px=radius_px,
        ids_to_track=list(ids_to_track or []),
    )
    return DropletIntensityRecorder(source, detector, tracker, cfg).run(progress=progress)
