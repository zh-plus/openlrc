#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

"""Crash-safe JSON checkpoint persistence."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def save_json_checkpoint(path: Path, data: dict) -> None:
    """Atomically replace *path* with a fully written JSON checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as file:
            temp_path = Path(file.name)
            json.dump(data, file, ensure_ascii=False, indent=4)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
