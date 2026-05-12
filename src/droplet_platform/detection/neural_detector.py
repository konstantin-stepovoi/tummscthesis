from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

try:  # optional dependency
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None

from .correlation_detector import DropletDetectorCuPy


if nn is not None:
    class TinyDropletNet(nn.Module):
        """Fallback architecture for heatmap-style droplet-net checkpoints.

        This mirrors the compact fully-convolutional network used during detector
        prototyping. If the saved `.pt` file contains a full TorchScript/module,
        it will be loaded directly; if it contains a state_dict, this architecture
        is used as the default target.
        """

        def __init__(self, in_channels: int = 1, hidden: int = 16):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, hidden, 5, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, hidden, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, hidden, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, 1, 1),
            )

        def forward(self, x):
            return self.net(x)
else:
    TinyDropletNet = None


@dataclass
class NeuralDetectorConfig:
    model_path: str | Path
    downscale_factor: int = 2
    threshold: float = 0.35
    min_distance: int = 40
    max_candidates: int = 20000
    device: str = "auto"  # auto | cpu | cuda
    normalize: bool = True
    apply_sigmoid: bool = True
    architecture_hidden: int = 16


class NeuralDropletDetector(DropletDetectorCuPy):
    """Detector with the same public API as DropletDetectorCuPy, but Torch heatmap backend.

    The class intentionally inherits the NMS helper and `detect_and_store_centers()`
    behavior from `DropletDetectorCuPy`, so existing notebook code can switch backend
    by replacing the detector instance only.
    """

    def __init__(self, cfg: NeuralDetectorConfig):
        if torch is None:
            raise RuntimeError("PyTorch is not available; install torch to use NeuralDropletDetector")
        self.cfg = cfg
        self.ds = int(cfg.downscale_factor)
        self.min_distance = int(cfg.min_distance)
        self.max_candidates = int(cfg.max_candidates)
        if cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.device)
        self.model = self._load_model(Path(cfg.model_path)).to(self.device)
        self.model.eval()

    def _load_model(self, path: Path):
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.nn.Module):
            return obj
        model = TinyDropletNet(hidden=int(self.cfg.architecture_hidden))
        if isinstance(obj, dict) and "state_dict" in obj:
            obj = obj["state_dict"]
        if isinstance(obj, dict):
            # Tolerate common DataParallel prefixes.
            obj = {k.replace("module.", ""): v for k, v in obj.items()}
            try:
                model.load_state_dict(obj, strict=False)
            except Exception as exc:
                raise RuntimeError(f"Could not load neural detector state_dict from {path}: {exc}") from exc
            return model
        raise RuntimeError(f"Unsupported Torch model file format: {path}")

    def _downscale_cpu(self, frame_gray: np.ndarray) -> np.ndarray:
        if self.ds == 1:
            return frame_gray.astype(np.float32, copy=False)
        h, w = frame_gray.shape[:2]
        return cv2.resize(frame_gray, (w // self.ds, h // self.ds), interpolation=cv2.INTER_AREA).astype(np.float32)

    def _heatmap(self, frame_gray: np.ndarray) -> np.ndarray:
        img = self._downscale_cpu(frame_gray)
        if self.cfg.normalize:
            img = (img - float(img.mean())) / (float(img.std()) + 1e-6)
        x = torch.from_numpy(img[None, None].astype(np.float32)).to(self.device)
        with torch.no_grad():
            y = self.model(x)
            if self.cfg.apply_sigmoid:
                y = torch.sigmoid(y)
        hm = y.detach().float().cpu().numpy()[0, 0]
        return hm.astype(np.float32, copy=False)

    @staticmethod
    def _local_maxima_numpy(hm: np.ndarray, min_distance: int, threshold: float, max_candidates: int) -> Tuple[np.ndarray, np.ndarray]:
        from scipy.ndimage import maximum_filter
        win = 2 * int(min_distance) + 1
        maxf = maximum_filter(hm, size=win, mode="constant", cval=-np.inf)
        cand = (hm >= maxf - 1e-6) & (hm > float(threshold))
        ys, xs = np.nonzero(cand)
        if ys.size == 0:
            return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float32)
        scores = hm[ys, xs].astype(np.float32)
        if ys.size > max_candidates:
            idx = np.argpartition(scores, -max_candidates)[-max_candidates:]
            ys, xs, scores = ys[idx], xs[idx], scores[idx]
        yx = np.stack([ys, xs], axis=1).astype(np.int32)
        centers = DropletDetectorCuPy._greedy_nms_grid(yx, scores, int(min_distance))
        score_map = {(int(y), int(x)): float(s) for (y, x), s in zip(yx, scores)}
        out_scores = np.asarray([score_map.get((int(y), int(x)), 0.0) for y, x in centers], dtype=np.float32)
        return centers, out_scores

    def detect_centers(self, frame_gray: np.ndarray, exclude_border: bool = False, return_scores: bool = False):
        hm = self._heatmap(frame_gray)
        centers, scores = self._local_maxima_numpy(
            hm,
            min_distance=max(1, self.min_distance // max(1, self.ds)),
            threshold=float(self.cfg.threshold),
            max_candidates=self.max_candidates,
        )
        if exclude_border and centers.size:
            md = max(1, self.min_distance // max(1, self.ds))
            h, w = hm.shape
            m = (centers[:, 0] >= md) & (centers[:, 0] < h - md) & (centers[:, 1] >= md) & (centers[:, 1] < w - md)
            centers, scores = centers[m], scores[m]
        if return_scores:
            return centers, scores
        return centers


def make_neural_detector(model_path: str | Path, **kwargs) -> NeuralDropletDetector:
    return NeuralDropletDetector(NeuralDetectorConfig(model_path=model_path, **kwargs))
