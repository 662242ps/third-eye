"""Configurable real-world object heights used by distance estimation."""

import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_DIR = PROJECT_ROOT / "settings"
OBJECT_HEIGHTS_FILE = SETTINGS_DIR / "object_heights.json"

# Approximate default heights in metres. These are starting points only;
# camera angle, object type and detector box quality still affect accuracy.
DEFAULT_OBJECT_HEIGHTS = {
    "person": 1.65,
    "car": 1.50,
    "truck": 2.80,
    "motorcycle": 1.20,
    "bus": 3.10,
}


def _normalize(data):
    """Return safe, complete calibration values in metres."""
    if not isinstance(data, dict):
        return DEFAULT_OBJECT_HEIGHTS.copy()

    values = {}
    for label, default in DEFAULT_OBJECT_HEIGHTS.items():
        try:
            value = float(data.get(label, default))
        except (TypeError, ValueError):
            value = default
        if not math.isfinite(value) or value < 0.1 or value > 10.0:
            value = default
        values[label] = round(value, 3)
    return values


def load_object_heights():
    if not OBJECT_HEIGHTS_FILE.is_file():
        values = DEFAULT_OBJECT_HEIGHTS.copy()
        try:
            save_object_heights(values)
        except OSError:
            pass
        return values

    try:
        with OBJECT_HEIGHTS_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        values = DEFAULT_OBJECT_HEIGHTS.copy()
        try:
            save_object_heights(values)
        except OSError:
            pass
        return values

    values = _normalize(data)
    if values != data:
        try:
            save_object_heights(values)
        except OSError:
            pass
    return values


def save_object_heights(values):
    normalized = _normalize(values)
    SETTINGS_DIR.mkdir(exist_ok=True)
    with OBJECT_HEIGHTS_FILE.open("w", encoding="utf-8") as file:
        json.dump(normalized, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return normalized
