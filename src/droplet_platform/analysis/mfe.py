from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class MFEConfig:
    fps: float
    period_s: float
    on_duration_s: float
    off_duration_s: float | None = None
    initial_phase_s: float = 0.0
    highpass_window_s: float | None = None
    min_valid_fraction: float = 0.5


def read_intensity_tsv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def normalize_trace_exponential(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Simple robust baseline removal for photobleaching-like trends.

    This intentionally avoids heavy dependencies. It fits log(y) = a + b t on
    positive finite values and returns y / baseline.
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    out = np.full_like(y, np.nan, dtype=float)
    m = np.isfinite(t) & np.isfinite(y) & (y > 0)
    if m.sum() < 5:
        return out
    coeff = np.polyfit(t[m], np.log(y[m]), 1)
    baseline = np.exp(coeff[1] + coeff[0] * t)
    out[m] = y[m] / baseline[m]
    return out


def rolling_highpass(y: np.ndarray, window: int) -> np.ndarray:
    if window is None or window <= 2:
        return y
    s = pd.Series(y)
    baseline = s.rolling(window=window, center=True, min_periods=max(2, window // 4)).median().to_numpy()
    return y - baseline + np.nanmedian(baseline)


def cycle_delta_i_over_i(t: np.ndarray, y_norm: np.ndarray, cfg: MFEConfig) -> list[float]:
    period = float(cfg.period_s)
    on_d = float(cfg.on_duration_s)
    off_d = float(cfg.off_duration_s if cfg.off_duration_s is not None else period - on_d)
    phase = float(cfg.initial_phase_s)
    if period <= 0 or on_d <= 0:
        raise ValueError("period_s and on_duration_s must be positive")
    values = []
    start = np.nanmin(t) + phase
    end = np.nanmax(t)
    c = start
    while c + period <= end:
        on = (t >= c) & (t < c + on_d)
        off = (t >= c + on_d) & (t < c + on_d + off_d)
        if np.isfinite(y_norm[on]).sum() >= 2 and np.isfinite(y_norm[off]).sum() >= 2:
            ion = np.nanmedian(y_norm[on])
            ioff = np.nanmedian(y_norm[off])
            if np.isfinite(ion) and np.isfinite(ioff) and abs(ioff) > 1e-12:
                values.append((ion - ioff) / ioff)
        c += period
    return values


def summarize_mfe_table(df: pd.DataFrame, cfg: MFEConfig) -> pd.DataFrame:
    """Compute droplet-level MFE summary from a TSV-like dataframe."""
    first = df.columns[0]
    timestamps = df[first].to_numpy(dtype=float)
    # If pypylon timestamps are ns-ish, use relative seconds. Otherwise assume seconds.
    if np.nanmax(timestamps) > 1e6:
        t = (timestamps - timestamps[0]) * 1e-9
    else:
        t = timestamps - timestamps[0]
    rows = []
    hp_window = None
    if cfg.highpass_window_s:
        hp_window = int(round(cfg.highpass_window_s * cfg.fps))
    for col in df.columns[1:]:
        y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        valid_fraction = np.isfinite(y).mean()
        if valid_fraction < cfg.min_valid_fraction:
            continue
        yn = normalize_trace_exponential(t, y)
        if hp_window:
            yn = rolling_highpass(yn, hp_window)
        cycles = np.asarray(cycle_delta_i_over_i(t, yn, cfg), dtype=float)
        if cycles.size == 0:
            continue
        rows.append({
            "id": col,
            "n_cycles": int(cycles.size),
            "deltaI_over_I": float(np.nanmedian(cycles)),
            "deltaI_over_I_pct": float(100.0 * np.nanmedian(cycles)),
            "cycle_std": float(np.nanstd(cycles)),
            "median_I_raw": float(np.nanmedian(y)),
            "valid_fraction": float(valid_fraction),
        })
    return pd.DataFrame(rows)
