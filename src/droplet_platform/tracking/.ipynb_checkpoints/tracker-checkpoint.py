# Tracker.py
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import threading
from typing import Dict, Tuple, Optional


@dataclass
class Track:
    id: int
    pos: np.ndarray            # shape (2,) float32, (y,x) full-res
    vel: np.ndarray            # shape (2,) float32, (dy,dx) per frame
    last_frame: int
    age: int = 0               # frames since last match
    hits: int = 1              # total matches
    score_ema: float = 0.0
    history: list = field(default_factory=list)  # optional, keep small if you want

    def predict(self, frame_id: int) -> np.ndarray:
        dt = max(1, int(frame_id - self.last_frame))
        return self.pos + self.vel * dt


class BubbleTracker:
    """
    Minimal online tracker:
      - Greedy matching by distance to predicted position (constant-velocity model)
      - Track birth for unmatched detections
      - Track death after max_age
    """

    def __init__(
        self,
        max_dist_px: float = 250.0,  # gate (full-res px)
        max_age: int = 5,            # how many frames a track can miss
        vel_alpha: float = 0.35,     # EMA for velocity update
        score_alpha: float = 0.2,    # EMA for score
        min_score_to_start: float = -np.inf,  # optional gate for new tracks
        prefer_direction: Optional[Tuple[float, float]] = None,  # (dy,dx) unit-ish
        direction_weight: float = 0.0,  # 0 disables, >0 penalizes wrong direction
    ):
        self.max_dist_px = float(max_dist_px)
        self.max_age = int(max_age)
        self.vel_alpha = float(vel_alpha)
        self.score_alpha = float(score_alpha)
        self.min_score_to_start = float(min_score_to_start)

        self.prefer_direction = None if prefer_direction is None else np.array(prefer_direction, dtype=np.float32)
        self.direction_weight = float(direction_weight)

        self._next_id = 1
        self.tracks: Dict[int, Track] = {}

        # if you ever call from multiple threads:
        self._lock = threading.Lock()

    def reset(self):
        with self._lock:
            self._next_id = 1
            self.tracks.clear()

    def _new_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    def update(self, frame_id: int, centers_yx: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """
        centers_yx: (N,2) int/float (y,x) full-res
        scores: (N,) float
        returns:
          layer (M,5) float32: [frame, id, y, x, score]
          where M == number of matched + started tracks for this frame
        """
        centers = np.asarray(centers_yx)
        if centers.size == 0:
            centers = centers.reshape(0, 2)
        centers = centers.astype(np.float32, copy=False)

        scores = np.asarray(scores, dtype=np.float32)
        if scores.size == 0:
            scores = scores.reshape(0,)

        with self._lock:
            # 1) Predict existing tracks
            track_ids = list(self.tracks.keys())
            T = len(track_ids)
            N = centers.shape[0]

            pred = np.empty((T, 2), dtype=np.float32)
            last_pos = np.empty((T, 2), dtype=np.float32)
            last_frame = np.empty((T,), dtype=np.int32)

            for i, tid in enumerate(track_ids):
                tr = self.tracks[tid]
                pred[i] = tr.predict(frame_id)
                last_pos[i] = tr.pos
                last_frame[i] = tr.last_frame

            # 2) Build cost matrix (distance + optional direction penalty), apply gating
            # If no tracks or no detections -> handle separately
            assignments = []  # list of (ti, dj, cost)

            if T > 0 and N > 0:
                # squared distance
                # dist2 shape (T,N)
                dy = pred[:, None, 0] - centers[None, :, 0]
                dx = pred[:, None, 1] - centers[None, :, 1]
                dist2 = dy * dy + dx * dx

                gate2 = (self.max_dist_px ** 2)
                valid = dist2 <= gate2

                cost = dist2.copy()

                # Optional: direction preference (penalize movement opposite to expected flow)
                if self.prefer_direction is not None and self.direction_weight > 0:
                    # expected direction vector (dy,dx)
                    d = self.prefer_direction
                    dn = float(np.sqrt(d[0]*d[0] + d[1]*d[1]) + 1e-6)
                    d = d / dn

                    # observed displacement from last_pos to detection
                    ody = centers[None, :, 0] - last_pos[:, None, 0]
                    odx = centers[None, :, 1] - last_pos[:, None, 1]
                    on = np.sqrt(ody*ody + odx*odx) + 1e-6
                    ody_n = ody / on
                    odx_n = odx / on
                    # cosine similarity with preferred direction
                    cos = ody_n * d[0] + odx_n * d[1]
                    # penalty if cos is low/negative (range roughly [0..2])
                    penalty = (1.0 - cos)
                    cost = cost + self.direction_weight * penalty * gate2

                cost[~valid] = np.inf

                # Greedy assignment by minimal cost
                # Complexity O(T*N) to sort all finite pairs; N~80 ok.
                ti, dj = np.where(np.isfinite(cost))
                if ti.size > 0:
                    cvals = cost[ti, dj]
                    order = np.argsort(cvals)
                    used_t = np.zeros((T,), dtype=bool)
                    used_d = np.zeros((N,), dtype=bool)
                    for k in order:
                        t = int(ti[k]); d = int(dj[k])
                        if used_t[t] or used_d[d]:
                            continue
                        used_t[t] = True
                        used_d[d] = True
                        assignments.append((t, d, float(cvals[k])))

                assigned_tracks = {t for (t, _, _) in assignments}
                assigned_dets = {d for (_, d, _) in assignments}
            else:
                assigned_tracks = set()
                assigned_dets = set()

            # 3) Update matched tracks
            out_rows = []

            for (t, d, _) in assignments:
                tid = track_ids[t]
                tr = self.tracks[tid]
                z = centers[d]
                s = float(scores[d]) if d < scores.size else 0.0

                dt = max(1, int(frame_id - tr.last_frame))
                # measured velocity
                v_meas = (z - tr.pos) / dt
                tr.vel = (1.0 - self.vel_alpha) * tr.vel + self.vel_alpha * v_meas
                tr.pos = z
                tr.last_frame = int(frame_id)
                tr.age = 0
                tr.hits += 1

                if tr.hits == 1:
                    tr.score_ema = s
                else:
                    tr.score_ema = (1.0 - self.score_alpha) * tr.score_ema + self.score_alpha * s

                # history optional
                # tr.history.append((frame_id, float(z[0]), float(z[1]), s))

                out_rows.append((float(frame_id), float(tid), float(z[0]), float(z[1]), float(s)))

            # 4) Age unmatched tracks, delete old
            to_del = []
            for tid in track_ids:
                tr = self.tracks[tid]
                # if not updated this frame
                if tr.last_frame != frame_id:
                    tr.age += 1
                    if tr.age > self.max_age:
                        to_del.append(tid)
            for tid in to_del:
                del self.tracks[tid]

            # 5) Create tracks for unmatched detections
            if centers.shape[0] > 0:
                for d in range(centers.shape[0]):
                    if d in assigned_dets:
                        continue
                    s = float(scores[d]) if d < scores.size else 0.0
                    if s < self.min_score_to_start:
                        continue

                    tid = self._new_id()
                    z = centers[d]
                    tr = Track(
                        id=tid,
                        pos=z.copy(),
                        vel=np.zeros((2,), dtype=np.float32),
                        last_frame=int(frame_id),
                        age=0,
                        hits=1,
                        score_ema=s,
                    )
                    self.tracks[tid] = tr
                    out_rows.append((float(frame_id), float(tid), float(z[0]), float(z[1]), float(s)))

            if len(out_rows) == 0:
                return np.zeros((0, 5), dtype=np.float32)

            return np.asarray(out_rows, dtype=np.float32)


# -------------------------
# Hook for your GUI
# -------------------------

# Create a default global tracker instance
_tracker = BubbleTracker(
    max_dist_px=400.0,     # <-- подстрой под твой поток/насос
    max_age=20,
    vel_alpha=0.35,
    score_alpha=0.2,
    min_score_to_start=-np.inf,
    prefer_direction=None, # например (0, +1) если поток вправо (y,x)
    direction_weight=0.0,
)

# Store last output (optional, for debugging/overlay later)
_last_tracks_layer = None

def tracker_hook(frame_id: int, centers_yx: np.ndarray, scores: np.ndarray):
    """
    Called from preview thread. Updates tracker state.
    Keeps last result in module global.
    """
    global _last_tracks_layer
    _last_tracks_layer = _tracker.update(frame_id, centers_yx, scores)
    # пока ничего не возвращаем
    # return _last_tracks_layer

def get_last_tracks_layer():
    return _last_tracks_layer

def reset_tracker():
    _tracker.reset()


def export_tracker_state() -> dict:
    """
    Снапшот трекера: можно хранить в STATE и восстанавливать после рестарта стрима.
    """
    with _tracker._lock:
        tracks = {}
        for tid, tr in _tracker.tracks.items():
            tracks[int(tid)] = {
                "id": int(tr.id),
                "pos": tr.pos.astype(np.float32).copy(),
                "vel": tr.vel.astype(np.float32).copy(),
                "last_frame": int(tr.last_frame),
                "age": int(tr.age),
                "hits": int(tr.hits),
                "score_ema": float(tr.score_ema),
            }
        return {
            "next_id": int(_tracker._next_id),
            "tracks": tracks,
        }

def import_tracker_state(state: dict):
    """
    Восстановление трекера из снапшота.
    """
    if not state:
        return
    with _tracker._lock:
        _tracker._next_id = int(state.get("next_id", 1))
        _tracker.tracks.clear()
        tracks = state.get("tracks", {})
        for tid, d in tracks.items():
            tid = int(tid)
            tr = Track(
                id=int(d["id"]),
                pos=np.asarray(d["pos"], dtype=np.float32).copy(),
                vel=np.asarray(d["vel"], dtype=np.float32).copy(),
                last_frame=int(d["last_frame"]),
                age=int(d.get("age", 0)),
                hits=int(d.get("hits", 1)),
                score_ema=float(d.get("score_ema", 0.0)),
            )
            _tracker.tracks[tid] = tr

def get_tracker_max_frame() -> int:
    """
    Чтобы GUI мог понять, откуда продолжать frame_id.
    """
    with _tracker._lock:
        if not _tracker.tracks:
            return -1
        return max(tr.last_frame for tr in _tracker.tracks.values())


class TrackerService:
    """Small stateful adapter around BubbleTracker.

    The legacy module exposes global functions. New code should prefer this class so
    UI, preview, and recording can own independent tracker instances.
    """

    def __init__(self, tracker: BubbleTracker | None = None):
        self.tracker = tracker or BubbleTracker()
        self.last_tracks_layer: np.ndarray | None = None

    def update(self, frame_id: int, centers_yx: np.ndarray, scores: np.ndarray) -> np.ndarray:
        self.last_tracks_layer = self.tracker.update(frame_id, centers_yx, scores)
        return self.last_tracks_layer

    def reset(self) -> None:
        self.tracker.reset()
        self.last_tracks_layer = None

    def ids(self) -> list[int]:
        arr = np.asarray(self.last_tracks_layer) if self.last_tracks_layer is not None else np.zeros((0, 5))
        if arr.size == 0 or arr.ndim != 2 or arr.shape[1] < 2:
            return []
        ids = np.unique(arr[:, 1].astype(np.int32, copy=False))
        return sorted(int(x) for x in ids if x > 0)
