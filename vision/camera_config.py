import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_DIR = PROJECT_ROOT / "settings"
CAMERA_SETTINGS_FILE = SETTINGS_DIR / "camera_settings.json"

DEFAULT_CAMERA_SETTINGS = {
    "camera_index": 0,
    "focal_length": 600.0,
}


def _normalize(data):
    try:
        camera_index = max(0, int(data.get("camera_index", 0)))
        focal_length = float(data.get("focal_length", DEFAULT_CAMERA_SETTINGS["focal_length"]))
    except (TypeError, ValueError, AttributeError):
        return DEFAULT_CAMERA_SETTINGS.copy()
    if not math.isfinite(focal_length) or focal_length <= 0:
        focal_length = DEFAULT_CAMERA_SETTINGS["focal_length"]
    return {
        "camera_index": camera_index,
        "focal_length": round(focal_length, 2),
    }


def load_camera_settings():
    if not CAMERA_SETTINGS_FILE.is_file():
        return DEFAULT_CAMERA_SETTINGS.copy()

    try:
        with CAMERA_SETTINGS_FILE.open("r", encoding="utf-8") as file:
            return _normalize(json.load(file))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_CAMERA_SETTINGS.copy()


def estimate_focal_length(distance_m, box_height_px, object_height_m):
    """Estimate focal length from one measured object in a frame.

    The pinhole-camera relation used by the detector is
    ``distance = real_height * focal / box_height``.
    """
    try:
        distance_m = float(distance_m)
        box_height_px = float(box_height_px)
        object_height_m = float(object_height_m)
    except (TypeError, ValueError):
        raise ValueError("Calibration values must be numeric")
    if not all(
        math.isfinite(value) and value > 0
        for value in (distance_m, box_height_px, object_height_m)
    ):
        raise ValueError("Calibration values must be greater than zero")
    return round(distance_m * box_height_px / object_height_m, 2)


def save_camera_settings(camera_index, focal_length):
    settings = _normalize(
        {"camera_index": camera_index, "focal_length": focal_length}
    )
    SETTINGS_DIR.mkdir(exist_ok=True)
    with CAMERA_SETTINGS_FILE.open("w", encoding="utf-8") as file:
        json.dump(settings, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return settings
