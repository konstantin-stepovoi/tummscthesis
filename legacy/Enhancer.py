# Enhancer.py
# Декоратор поверх DropletDetectorCuPy: 2-pass детекция + опциональный preprocess кадра.
# Ничего не меняем в droplet_lib.py, просто оборачиваем готовый детектор.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Any

import numpy as np
import cv2


@dataclass
class EnhancerConfig:
    enabled: bool = True

    # --- Preprocess (CPU) ---
    preprocess: str = "none"   # "none" | "highpass" | "clahe"
    hp_sigma: float = 10.0     # для highpass (в пикселях full-res!)
    clahe_clip: float = 2.0
    clahe_grid: Tuple[int, int] = (8, 8)

    # --- Second pass logic ---
    second_pass: bool = True
    second_pass_percentile: float = 84.0
    # когда запускать второй проход:
    trigger_min_count: int = 0          # если detections < trigger_min_count -> делаем 2-й проход
    trigger_min_score: Optional[float] = None  # если median(score) < ... -> делаем 2-й проход
    always_second_pass: bool = False    # принудительно всегда делать 2-й проход

    # ограничение кандидатов на 2-м проходе (если хочешь отдельно)
    second_pass_max_candidates: Optional[int] = None


class EnhancedDetector:
    """
    Wrapper/decorator.

    Важно:
    - __getattr__ делегирует всё на base, кроме detect_centers().
    - detect_and_store_centers()/detect_frame_to_dataset_layer работают автоматически,
      т.к. они вызывают detector.detect_centers().
    - Внутри detect_centers мы временно трогаем base.percentile (и опционально max_candidates)
      ТОЛЬКО на время второго вызова. Это НЕ thread-safe, если один base используется одновременно
      из нескольких потоков. В твоём пайплайне детектор обычно живёт в одном детектор-треде — ок.
    """
    def __init__(self, base_detector, cfg: EnhancerConfig):
        self.base = base_detector
        self.cfg = cfg

        # CLAHE object можно закэшировать
        self._clahe = None
        if self.cfg.preprocess == "clahe":
            self._clahe = cv2.createCLAHE(
                clipLimit=float(self.cfg.clahe_clip),
                tileGridSize=tuple(self.cfg.clahe_grid),
            )

    def __getattr__(self, name: str) -> Any:
        # всё, чего нет в wrapper, берём у base
        return getattr(self.base, name)

    # -------------------------
    # Preprocess helpers
    # -------------------------
    def _preprocess_frame(self, frame_gray: np.ndarray) -> np.ndarray:
        if (not self.cfg.enabled) or (self.cfg.preprocess == "none"):
            return frame_gray

        if frame_gray.dtype != np.uint8:
            frame_gray = frame_gray.astype(np.uint8, copy=False)

        if self.cfg.preprocess == "clahe":
            # CLAHE обычно помогает с локальным контрастом
            return self._clahe.apply(frame_gray)

        if self.cfg.preprocess == "highpass":
            # High-pass: img - gaussian(img)
            # sigma в full-res! (т.к. preprocess до downscale)
            f = frame_gray.astype(np.float32)
            bg = cv2.GaussianBlur(f, ksize=(0, 0), sigmaX=float(self.cfg.hp_sigma), sigmaY=float(self.cfg.hp_sigma))
            hp = f - bg

            # привести обратно к uint8 (стабильно)
            mn = float(hp.min())
            mx = float(hp.max())
            hp = hp - mn
            if mx - mn > 1e-6:
                hp *= (255.0 / (mx - mn))
            hp = np.clip(hp, 0, 255).astype(np.uint8)
            return hp

        # неизвестный режим — не трогаем
        return frame_gray

    # -------------------------
    # Merge helpers
    # -------------------------
    @staticmethod
    def _scores_by_coord(centers: np.ndarray, scores: np.ndarray) -> dict:
        # dict[(y,x)] = max score
        d = {}
        for (y, x), s in zip(centers, scores):
            key = (int(y), int(x))
            sv = float(s)
            prev = d.get(key)
            if prev is None or sv > prev:
                d[key] = sv
        return d

    def _merge_and_nms(
        self,
        c1: np.ndarray, s1: np.ndarray,
        c2: np.ndarray, s2: np.ndarray,
        min_distance: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        if c1.size == 0:
            return c2.astype(np.int32, copy=False), s2.astype(np.float32, copy=False)
        if c2.size == 0:
            return c1.astype(np.int32, copy=False), s1.astype(np.float32, copy=False)

        yx = np.vstack([c1, c2]).astype(np.int32, copy=False)
        sc = np.concatenate([s1, s2]).astype(np.float32, copy=False)

        # Используем уже имеющуюся реализацию NMS из base
        centers = self.base._greedy_nms_grid(yx, sc, int(min_distance))
        if centers.size == 0:
            return centers, np.empty((0,), dtype=np.float32)

        # восстановим score для выбранных координат (по точному совпадению координат)
        d = self._scores_by_coord(yx, sc)
        out_scores = np.asarray([d.get((int(y), int(x)), 0.0) for (y, x) in centers], dtype=np.float32)
        return centers, out_scores

    # -------------------------
    # Main override
    # -------------------------
    def detect_centers(self, frame_gray: np.ndarray, exclude_border: bool = False, return_scores: bool = False):
        """
        API совместим с твоей версией: возвращаем либо centers, либо (centers, scores)
        """
        if not self.cfg.enabled:
            return self.base.detect_centers(frame_gray, exclude_border=exclude_border, return_scores=return_scores)

        frame_in = self._preprocess_frame(frame_gray)

        # 1) базовый проход
        c1, s1 = self.base.detect_centers(frame_in, exclude_border=exclude_border, return_scores=True)

        if not self.cfg.second_pass:
            if return_scores:
                return c1, s1
            return c1

        # Решаем, нужен ли второй проход
        need2 = bool(self.cfg.always_second_pass)

        if (not need2) and (self.cfg.trigger_min_count > 0):
            if c1.shape[0] < int(self.cfg.trigger_min_count):
                need2 = True

        if (not need2) and (self.cfg.trigger_min_score is not None):
            if s1.size == 0:
                need2 = True
            else:
                med = float(np.median(s1))
                if med < float(self.cfg.trigger_min_score):
                    need2 = True

        if not need2:
            if return_scores:
                return c1, s1
            return c1

        # 2) второй проход: временно уменьшаем percentile (и опционально max_candidates)
        old_pct = getattr(self.base, "percentile", None)
        old_mc = getattr(self.base, "max_candidates", None)

        try:
            self.base.percentile = float(self.cfg.second_pass_percentile)
            if self.cfg.second_pass_max_candidates is not None:
                self.base.max_candidates = int(self.cfg.second_pass_max_candidates)

            c2, s2 = self.base.detect_centers(frame_in, exclude_border=exclude_border, return_scores=True)

        finally:
            # вернуть настройки обратно
            if old_pct is not None:
                self.base.percentile = old_pct
            if old_mc is not None and self.cfg.second_pass_max_candidates is not None:
                self.base.max_candidates = old_mc

        # мердж + финальный NMS
        centers, scores = self._merge_and_nms(c1, s1, c2, s2, min_distance=int(self.base.min_distance))

        if return_scores:
            return centers, scores
        return centers


def enhance_detector(base_detector, **cfg_kwargs) -> EnhancedDetector:
    """
    Удобный factory:
      det = enhance_detector(base, preprocess="highpass", hp_sigma=10.0, second_pass_percentile=84.0, trigger_min_count=70)
    """
    cfg = EnhancerConfig(**cfg_kwargs)
    return EnhancedDetector(base_detector, cfg)
