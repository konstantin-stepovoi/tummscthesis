import time
import datetime
import traceback
import threading
import queue
from dataclasses import dataclass, field
from pathlib import Path
from collections import deque

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import ipywidgets as widgets
from IPython.display import display

import winsound

from pypylon import pylon

def precompute_disk_offsets(radius: int) -> np.ndarray:
    """
    Возвращает массив оффсетов (dy,dx) для пикселей в диске радиуса radius.
    """
    r = int(radius)
    ys, xs = np.ogrid[-r:r+1, -r:r+1]
    mask = (ys*ys + xs*xs) <= (r*r)
    dy, dx = np.where(mask)
    dy = dy - r
    dx = dx - r
    return np.stack([dy, dx], axis=1).astype(np.int16)

def sum_in_disk_u8(img_u8: np.ndarray, cy: float, cx: float, offsets: np.ndarray) -> int:
    """
    Быстро суммирует интенсивность в диске вокруг (cy,cx) на mono8 изображении.
    Возвращает int.
    """
    H, W = img_u8.shape
    y0 = int(round(cy))
    x0 = int(round(cx))

    yy = y0 + offsets[:, 0].astype(np.int32)
    xx = x0 + offsets[:, 1].astype(np.int32)

    m = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
    if not np.any(m):
        return 0
    return int(img_u8[yy[m], xx[m]].sum(dtype=np.int64))

def get_ids_from_last_annotation() -> list[int]:
    tl = STATE.last_tracks_layer
    if tl is None:
        return []
    arr = np.asarray(tl)
    if arr.size == 0 or arr.ndim != 2 or arr.shape[1] < 4:
        return []
    ids = np.unique(arr[:, 1].astype(np.int32, copy=False))
    ids = [int(x) for x in ids if x > 0]
    ids.sort()
    return ids

def record_sequence_bubbles_intensity(
    out_file: Path,
    fps: float,
    exposure_ms: float,
    duration_s: float,
    detector,  # DropletDetectorCuPy (инициализируешь заранее)
    *,
    radius_px: int,
    ids_to_track: list[int] | None = None,
    show_live: bool = True,
    colorize: bool = True,
    timeout_ms: int = 5000,
):
    """
    Пишет TSV:
      col0: timestamp_ns
      далее: по одному столбцу на bubble ID (header = id)
    Интенсивность = сумма пикселей mono8 внутри круга radius_px вокруг позиции трека.

    Важно: предполагается, что Tracker уже импортирован и hook/инстанс живёт,
    т.е. tracker_hook(frame_id, centers, scores) и get_last_tracks_layer() доступны.
    """

    # 1) какие ID меряем
    if ids_to_track is None:
        ids_to_track = get_ids_from_last_annotation()
    ids_to_track = [int(x) for x in ids_to_track if int(x) > 0]
    ids_to_track.sort()

    if len(ids_to_track) == 0:
        raise RuntimeError("ids_to_track is empty. Сначала закрой стрим и убедись, что есть аннотации (STATE.last_tracks_layer).")

    # 2) готовим диск
    offsets = precompute_disk_offsets(int(radius_px))

    # 3) камера
    cam, conv = make_camera()
    camera_configure(cam, fps=fps, exposure_ms=exposure_ms)

    # 4) подготовка файла
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    header = ["timestamp_ns"] + [str(i) for i in ids_to_track]

    win = "Bubble intensity (REC) (q=stop)"
    if show_live:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1000, 700)

    n_frames = int(duration_s * fps)

    # 5) запись
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")

        try:
            cam.StartGrabbing()
            t0 = time.time()
            print(f"▶ Bubble-intensity recording: {n_frames} frames @ {fps:g} fps | ids={len(ids_to_track)}")

            for frame_id in range(n_frames):
                grab = cam.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)
                try:
                    if not grab.GrabSucceeded():
                        continue

                    gray = conv.Convert(grab).GetArray()  # mono8
                    ts = int(grab.TimeStamp)              # ns-ish from pylon (как у тебя)

                    # --- детекция + трекинг (синхронно, чтобы мерить корректно) ---
                    layer = detect_frame_to_dataset_layer(gray, frame_id, detector)  # (N,4): fid,y,x,score
                    if layer.shape[0] == 0:
                        centers = np.empty((0, 2), dtype=np.int32)
                        scores = np.empty((0,), dtype=np.float32)
                    else:
                        centers = layer[:, 1:3].astype(np.int32, copy=False)  # (y,x)
                        scores  = layer[:, 3].astype(np.float32, copy=False)

                    tracker_hook(frame_id, centers, scores)
                    tracks_layer = get_last_tracks_layer()  # (M,5): [frame,id,y,x,score]

                    # --- быстрый индекс: id -> (y,x) на этом кадре ---
                    pos = {}
                    arr = np.asarray(tracks_layer) if tracks_layer is not None else np.zeros((0, 5), dtype=np.float32)
                    if arr.size and arr.ndim == 2 and arr.shape[1] >= 4:
                        # столбец id
                        ids_now = arr[:, 1].astype(np.int32, copy=False)
                        for rid in ids_to_track:
                            m = (ids_now == rid)
                            if np.any(m):
                                row = arr[m][0]
                                pos[rid] = (float(row[2]), float(row[3]))  # (y,x)

                    # --- измерения ---
                    row_vals = []
                    for rid in ids_to_track:
                        if rid in pos:
                            y, x = pos[rid]
                            s = sum_in_disk_u8(gray, y, x, offsets)
                            row_vals.append(str(s))
                        else:
                            row_vals.append("nan")

                    f.write(str(ts) + "\t" + "\t".join(row_vals) + "\n")

                    # --- опционально показываем ---
                    if show_live:
                        disp = make_disp(gray, bool(colorize))
                        # рисуем только tracked IDs
                        if arr.size:
                            # отрисуем треки цветом + ID
                            disp = draw_tracks_on_frame(disp, arr, favorite_id=(STATE.favorite_id if STATE.favorite_enabled else None))
                        cv2.imshow(win, disp)
                        if (cv2.waitKey(1) & 0xFF) == ord("q"):
                            break

                finally:
                    grab.Release()

        except Exception:
            traceback.print_exc()
        finally:
            try:
                cam.StopGrabbing()
            except Exception:
                pass
            try:
                cam.Close()
            except Exception:
                pass
            if show_live:
                cv2.destroyAllWindows()
            try:
                winsound.Beep(1200, 250)
            except Exception:
                pass

    print(f"✅ Saved bubble intensity table: {out_file}")
    return out_file

"""
Пример вызова:
ids = get_ids_from_last_annotation()

out_path = Path(root_w.value.strip().strip('"')) / "bubble_intensity.tsv"

record_sequence_bubbles_intensity(
    out_file=out_path,
    fps=float(fps_w.value),
    exposure_ms=float(exposure_w.value),
    duration_s=float(duration_w.value),
    detector=DET.detector,                # или твой detector instance
    radius_px=int(det_radius_w.value),    # радиус как в детекторе
    ids_to_track=ids,
    show_live=True,
    colorize=bool(colorize_w.value),
)
"""
