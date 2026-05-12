import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import time
import cupy as cp
import cupyx.scipy.ndimage as cnd
import cupyx.scipy.signal as csignal

class DropletDetectorCuPy:
    """
    GPU pipeline (CuPy):
      - Downscale: CPU (cv2.resize) or GPU (block-mean)
      - GPU: normalize -> correlation (ndimage or FFT) -> percentile threshold
      - GPU: candidates via maximum_filter
      - CPU: greedy NMS + output centers
    """

    def __init__(
        self,
        percentile=89.0,
        downscale_factor=2,
        droplet_radius_px=100,
        min_distance=40,
        eps=1e-6,
        cpu_downscale=True,
        sample_percentile_stride=1,
        max_candidates=20000,
        corr_backend="ndimage",      # "ndimage" | "fft"
        peak_downscale=1,            # 1 = как было; 2/4 = делать peak detection на более грубой corr карте
        template_path=None,          # str | None (".npy" или картинка)
        template_array=None,         # np.ndarray | None (float32/uint8)
        template_resize="auto",      # "auto" | "none" | (h,w) tuple
        thr_mode="per_frame",        # "per_frame" | "cached" | "ema"
        thr_update_every=10,         # обновлять порог раз в N кадров (для cached/ema)
        thr_ema_alpha=0.2,           # EMA: new = (1-a)*old + a*new

    ):
        self.percentile = float(percentile)
        self.ds = int(downscale_factor)
        self.min_distance = int(min_distance)
        self.eps = float(eps)
        self.cpu_downscale = bool(cpu_downscale)
        self.sample_stride = int(sample_percentile_stride)
        self.max_candidates = int(max_candidates)
        self.corr_backend = str(corr_backend).lower()
        self.peak_downscale = int(peak_downscale)
        self.thr_mode = str(thr_mode)
        self.thr_update_every = int(thr_update_every)
        self.thr_ema_alpha = float(thr_ema_alpha)

        self._thr_cache = None          # python float
        self._thr_cache_shape = None    # (H,W) corr shape
        self._frame_counter = 0


        if self.corr_backend not in ("ndimage", "fft"):
            raise ValueError("corr_backend must be 'ndimage' or 'fft'")
        if self.peak_downscale < 1:
            raise ValueError("peak_downscale must be >= 1")

        # --- Template size implied by droplet radius (in downscaled grid) ---
        r = droplet_radius_px // self.ds
        self.r = int(r)
        self.tmpl_size = int(2 * self.r + 1)

        tmpl = self._prepare_template(
            template_path=template_path,
            template_array=template_array,
            template_resize=template_resize
        )
        self.tmpl_gpu = cp.asarray(tmpl, dtype=cp.float32)

        # Для FFT: conv(img, flip(tmpl)) == correlate(img, tmpl)
        self.tmpl_flip_gpu = self.tmpl_gpu[::-1, ::-1]

        # окно для maximum_filter на “основной” corr карте
        self.win = 2 * self.min_distance + 1

    def _downscale(self, frame_gray: np.ndarray) -> np.ndarray:
        """CPU downscale (INTER_AREA), output float32."""
        h, w = frame_gray.shape
        if self.ds == 1:
            return frame_gray.astype(np.float32, copy=False)

        frame_ds = cv2.resize(
            frame_gray,
            (w // self.ds, h // self.ds),
            interpolation=cv2.INTER_AREA
        )
        return frame_ds.astype(np.float32, copy=False)

    @staticmethod
    def _downscale_gpu_blockmean(img_u8_gpu: cp.ndarray, ds: int) -> cp.ndarray:
        """
        GPU downscale через block mean (очень быстро).
        Требует обрезки до кратного ds.
        """
        if ds == 1:
            return img_u8_gpu.astype(cp.float32)

        h, w = img_u8_gpu.shape
        h2 = (h // ds) * ds
        w2 = (w // ds) * ds
        img_u8_gpu = img_u8_gpu[:h2, :w2]
        # reshape -> mean по блокам ds×ds
        return img_u8_gpu.reshape(h2 // ds, ds, w2 // ds, ds).mean(axis=(1, 3), dtype=cp.float32)

    @staticmethod
    def _downscale_gpu_blockmean_any(img_gpu: cp.ndarray, factor: int) -> cp.ndarray:
        """
        Универсальный block-mean для float32/float64 corr карты.
        """
        if factor == 1:
            return img_gpu
        h, w = img_gpu.shape
        h2 = (h // factor) * factor
        w2 = (w // factor) * factor
        x = img_gpu[:h2, :w2]
        return x.reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3))

    @staticmethod
    def _greedy_nms_grid(yx: np.ndarray, scores: np.ndarray, min_distance: int) -> np.ndarray:
        if yx.shape[0] == 0:
            return yx

        order = np.argsort(scores)[::-1]
        yx = yx[order]

        cell = max(1, int(min_distance))
        min_d2 = min_distance * min_distance

        grid = {}
        out = []

        for y, x in yx:
            cy, cx = y // cell, x // cell
            ok = True
            for gy in (cy - 1, cy, cy + 1):
                for gx in (cx - 1, cx, cx + 1):
                    pts = grid.get((gy, gx))
                    if not pts:
                        continue
                    for py, px in pts:
                        dy = int(y - py)
                        dx = int(x - px)
                        if dy * dy + dx * dx < min_d2:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break

            if ok:
                out.append((int(y), int(x)))
                grid.setdefault((cy, cx), []).append((int(y), int(x)))

        return np.asarray(out, dtype=np.int32)

    def _correlate(self, img_gpu: cp.ndarray) -> cp.ndarray:
        """
        Возвращает corr (GPU) того же размера, что img_gpu.
        """
        if self.corr_backend == "ndimage":
            return cnd.correlate(img_gpu, self.tmpl_gpu, mode="constant", cval=0.0).astype(cp.float32)
        else:
            # fftconvolve выполняет свёртку -> используем flip(tmpl) для корреляции
            return csignal.fftconvolve(img_gpu, self.tmpl_flip_gpu, mode="same").astype(cp.float32)

    def _prepare_template(self, template_path=None, template_array=None, template_resize="auto"):
        """
        Возвращает float32 template, подготовленный для корреляции:
        zero-mean + L2 norm.

        template_path:
          - .npy: ожидается float32/float64 (любого размера)
          - image: png/jpg/tif — читаем как grayscale

        template_resize:
          - "auto": приводим к (self.tmpl_size, self.tmpl_size)
          - "none": оставляем как есть
          - (h,w): приводим к указанному
        """
        # 1) load
        tmpl = None
        if template_array is not None:
            tmpl = np.asarray(template_array)
        elif template_path is not None:
            ext = os.path.splitext(str(template_path))[1].lower()
            if ext == ".npy":
                tmpl = np.load(template_path)
            else:
                img = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise RuntimeError(f"Cannot read template image: {template_path}")
                if img.ndim == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                tmpl = img
        else:
            # 2) default circular template (старое поведение)
            s = self.tmpl_size
            r = self.r
            yy, xx = np.ogrid[-r:r+1, -r:r+1]
            mask = (xx * xx + yy * yy) <= (r * r)
            tmpl = np.zeros((s, s), np.float32)
            tmpl[mask] = 1.0

        # 3) to float32
        tmpl = tmpl.astype(np.float32, copy=False)

        # 4) resize policy
        if template_resize == "auto":
            target = (self.tmpl_size, self.tmpl_size)
            if tmpl.shape != target:
                tmpl = cv2.resize(tmpl, (target[1], target[0]), interpolation=cv2.INTER_AREA)
        elif template_resize == "none":
            pass
        else:
            # tuple (h,w)
            th, tw = template_resize
            if tmpl.shape != (th, tw):
                tmpl = cv2.resize(tmpl, (tw, th), interpolation=cv2.INTER_AREA)

        # 5) normalize for correlation
        tmpl = tmpl - float(tmpl.mean())
        tmpl = tmpl / (float(np.linalg.norm(tmpl)) + self.eps)

        return tmpl


    def detect_centers(
        self,
        frame_gray: np.ndarray,
        exclude_border: bool = False,
        return_scores: bool = False
    ):
        """
        centers (N,2) в downscale-координатах [y,x].
        Если return_scores=True -> возвращает (centers, scores),
        где scores (N,) float32 — значение corr в точках centers.
        """
    
        # 1) Downscale -> img on GPU (float32)
        if self.cpu_downscale:
            frame_ds = self._downscale(frame_gray)  # np.float32
            img = cp.asarray(frame_ds, dtype=cp.float32)
        else:
            img_u8 = cp.asarray(frame_gray, dtype=cp.uint8)
            img = self._downscale_gpu_blockmean(img_u8, self.ds)
    
        # 2) Normalize on GPU
        img = (img - img.mean()) / (img.std() + self.eps)
    
        # 3) Correlation on GPU (ndimage or FFT)
        corr = self._correlate(img)
    
        # 4) Threshold percentile (cacheable)
        self._frame_counter += 1
        corr_shape = tuple(corr.shape)
    
        need_update = False
        if self.thr_mode == "per_frame":
            need_update = True
        else:
            if (self._thr_cache is None) or (self._thr_cache_shape != corr_shape):
                need_update = True
            elif self.thr_update_every > 0 and (self._frame_counter % self.thr_update_every == 0):
                need_update = True
    
        if need_update:
            if self.sample_stride > 1:
                corr_s = corr[::self.sample_stride, ::self.sample_stride]
                thr_gpu = cp.percentile(corr_s, self.percentile)
            else:
                thr_gpu = cp.percentile(corr, self.percentile)
    
            thr_new = float(thr_gpu.item())
            self._thr_cache_shape = corr_shape
    
            if self.thr_mode == "ema" and (self._thr_cache is not None):
                a = self.thr_ema_alpha
                self._thr_cache = (1.0 - a) * float(self._thr_cache) + a * thr_new
            else:
                self._thr_cache = thr_new
    
        thr = float(self._thr_cache)
    
        # --------------------------
        # 5) Peak detection on coarser corr map (optional)
        # --------------------------
        peak_ds = self.peak_downscale
        if peak_ds > 1:
            corr_peak = self._downscale_gpu_blockmean_any(corr, peak_ds)
            md_peak = max(1, self.min_distance // peak_ds)
            win_peak = 2 * md_peak + 1
    
            maxf = cnd.maximum_filter(corr_peak, size=win_peak, mode="constant", cval=-cp.inf)
            cand = (corr_peak >= (maxf - 1e-6)) & (corr_peak > thr)
    
            if exclude_border and md_peak > 0:
                cand[:md_peak, :] = False
                cand[-md_peak:, :] = False
                cand[:, :md_peak] = False
                cand[:, -md_peak:] = False
    
            ys, xs = cp.nonzero(cand)
            if ys.size == 0:
                if return_scores:
                    return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float32)
                return np.empty((0, 2), dtype=np.int32)
    
            scores = corr_peak[ys, xs]
            m = int(ys.size)
            if m > self.max_candidates:
                k = self.max_candidates
                idx = cp.argpartition(scores, -k)[-k:]
                ys, xs, scores = ys[idx], xs[idx], scores[idx]
    
            # CPU candidates for greedy NMS
            yx_cpu = cp.stack([ys, xs], axis=1).astype(cp.int32).get()
            scores_cpu = scores.astype(cp.float32).get()
    
            centers_peak = self._greedy_nms_grid(yx_cpu, scores_cpu, md_peak)  # coords on corr_peak grid
    
            if centers_peak.size == 0:
                if return_scores:
                    return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float32)
                return np.empty((0, 2), dtype=np.int32)
    
            # Convert centers to corr grid (multiply by peak_ds)
            centers = centers_peak.astype(np.int32, copy=False)
            centers[:, 0] *= peak_ds
            centers[:, 1] *= peak_ds
    
            if not return_scores:
                return centers
    
            # Важно: score хотим в координатах corr (а не corr_peak),
            # чтобы все режимы возвращали одинаковую "семантику" score.
            c_gpu = cp.asarray(centers, dtype=cp.int32)
            sc_gpu = corr[c_gpu[:, 0], c_gpu[:, 1]].astype(cp.float32)
            scores_out = sc_gpu.get()
    
            return centers, scores_out
    
        # --------------------------
        # 6) Standard peak detection on corr
        # --------------------------
        maxf = cnd.maximum_filter(corr, size=self.win, mode="constant", cval=-cp.inf)
        cand = (corr >= (maxf - 1e-6)) & (corr > thr)
    
        if exclude_border and self.min_distance > 0:
            md = self.min_distance
            cand[:md, :] = False
            cand[-md:, :] = False
            cand[:, :md] = False
            cand[:, -md:] = False
    
        ys, xs = cp.nonzero(cand)
        m = int(ys.size)
        if m == 0:
            if return_scores:
                return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float32)
            return np.empty((0, 2), dtype=np.int32)
    
        scores = corr[ys, xs]
        if m > self.max_candidates:
            k = self.max_candidates
            idx = cp.argpartition(scores, -k)[-k:]
            ys, xs, scores = ys[idx], xs[idx], scores[idx]
    
        yx_cpu = cp.stack([ys, xs], axis=1).astype(cp.int32).get()
        scores_cpu = scores.astype(cp.float32).get()
    
        centers = self._greedy_nms_grid(yx_cpu, scores_cpu, self.min_distance)
    
        if not return_scores:
            return centers
    
        if centers.size == 0:
            return centers, np.empty((0,), dtype=np.float32)
    
        # scores for FINAL centers (after NMS)
        c_gpu = cp.asarray(centers, dtype=cp.int32)
        sc_gpu = corr[c_gpu[:, 0], c_gpu[:, 1]].astype(cp.float32)
        scores_out = sc_gpu.get()
    
        return centers, scores_out


    def detect_and_store_centers(self, frame_gray, frame_idx, dataset, write_ptr, show=False):
        centers, scores = self.detect_centers(frame_gray, exclude_border=False, return_scores=True)
        n = centers.shape[0]
        if n == 0:
            return write_ptr

        end = write_ptr + n
        if end > dataset.shape[0]:
            n = max(0, dataset.shape[0] - write_ptr)
            centers = centers[:n]
            end = write_ptr + n
            if n == 0:
                return write_ptr

        dataset[write_ptr:end, 0] = frame_idx
        dataset[write_ptr:end, 1] = centers[:, 0] * self.ds
        dataset[write_ptr:end, 2] = centers[:, 1] * self.ds

        # score: лучше хранить float32 (и dataset должен быть float!)
        dataset[write_ptr:end, 3] = scores.astype(np.float32, copy=False)
        return end


def debug_one_frame_from_video(
    video_path: str,
    frame_number: int,
    detector,
    marker: str = "r+",
    figsize=(10, 5)
):
    """
    Берёт из видео кадр с номером frame_number (0-based),
    прогоняет через detector.detect_centers(),
    показывает картинку с маркерами.

    Возвращает: gray (H,W uint8), centers_full (N,2 int32) в координатах оригинала [y,x]
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # Попытка перемотки на нужный кадр
    ok = cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
    if not ok:
        # не критично — просто попробуем read, но предупреждаем
        print("Warning: CAP_PROP_POS_FRAMES set() returned False, will try reading anyway.")

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError(f"Cannot read frame {frame_number} from {video_path}")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Детекция (centers_ds: downscale coords)
    centers_ds = detector.detect_centers(gray, exclude_border=False)

    # Перевод в координаты оригинала
    ds = int(detector.ds)
    centers_full = centers_ds.copy()
    centers_full[:, 0] *= ds
    centers_full[:, 1] *= ds

    # --- Визуализация ---
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(gray, cmap="gray")
    if centers_full.shape[0] > 0:
       ax.plot(centers_full[:, 1], centers_full[:, 0], marker, markersize=8, linestyle="None")
    ax.set_title(f"Frame #{frame_number} | detected: {centers_full.shape[0]}")
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    return gray, centers_full


def run_video_processing_cuda(video_path, dataset, detector, max_frames=None, sync_every=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    write_ptr = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        write_ptr = detector.detect_and_store_centers(
            frame_gray=gray,
            frame_idx=frame_idx,
            dataset=dataset,
            write_ptr=write_ptr,
            show=False
        )

        frame_idx += 1

        # Для честного учёта GPU времени внутри профиля
        if sync_every and (frame_idx % sync_every == 0):
            cp.cuda.Stream.null.synchronize()

        if max_frames is not None and frame_idx >= max_frames:
            break

    # финальная синхронизация, чтобы все ядра завершились
    cp.cuda.Stream.null.synchronize()
    cap.release()
    return write_ptr

def detect_frame_to_dataset_layer(gray: np.ndarray, frame_idx: int, detector) -> np.ndarray:
    centers_ds, scores = detector.detect_centers(gray, exclude_border=False, return_scores=True)
    if centers_ds.size == 0:
        return np.zeros((0,4), dtype=np.float32)

    ds = int(detector.ds)
    centers_full = centers_ds.copy()
    centers_full[:,0] *= ds
    centers_full[:,1] *= ds

    out = np.empty((centers_full.shape[0], 4), dtype=np.float32)
    out[:,0] = float(frame_idx)
    out[:,1] = centers_full[:,0].astype(np.float32)
    out[:,2] = centers_full[:,1].astype(np.float32)
    out[:,3] = scores.astype(np.float32)
    return out
#================================
# =======================================================================