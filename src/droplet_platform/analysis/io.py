from __future__ import annotations

from pathlib import Path
import pandas as pd


def read_tsv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
