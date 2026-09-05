from __future__ import annotations

import shutil
import ssl
import urllib.error
import urllib.request
from pathlib import Path

import certifi

from config import POSE_MODEL_URL


def ensure_pose_model(path: Path) -> Path:
    """Download the official task bundle once; all pose inference remains local."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 100_000:
        return path
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        print("Downloading local MediaPipe pose model…")
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(POSE_MODEL_URL, timeout=30, context=context) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(path)
    except (OSError, urllib.error.URLError) as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download the local pose model: {error}") from error
    return path
