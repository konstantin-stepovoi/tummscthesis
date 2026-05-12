from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_mfe_distribution(summary: pd.DataFrame, column: str = "deltaI_over_I_pct"):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(summary[column].dropna(), bins=30)
    ax.set_xlabel(r"$\Delta I/I$ (%)")
    ax.set_ylabel("Count")
    ax.set_title("Droplet-level MFE metric distribution")
    ax.grid(alpha=0.3)
    return fig, ax
