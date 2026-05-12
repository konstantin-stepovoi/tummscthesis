from __future__ import annotations

from pathlib import Path
from datetime import datetime


def experiment_dir(root: str | Path, prefix: str = "exp") -> Path:
    p = Path(root) / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    p.mkdir(parents=True, exist_ok=True)
    return p
