import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_DIR = PROJECT_ROOT / "settings"
DISTANCE_SETTINGS_FILE = SETTINGS_DIR / "distance_thresholds.json"

DEFAULT_THRESHOLDS = {
    "danger": 5.0,
    "warning": 20.0,
}


def _normalize_thresholds(data):
    try:
        danger = float(data.get("danger", DEFAULT_THRESHOLDS["danger"]))
        warning = float(data.get("warning", DEFAULT_THRESHOLDS["warning"]))
    except (TypeError, ValueError, AttributeError):
        return DEFAULT_THRESHOLDS.copy()

    if danger <= 0:
        danger = DEFAULT_THRESHOLDS["danger"]
    if warning <= danger:
        warning = max(DEFAULT_THRESHOLDS["warning"], danger + 1.0)

    return {
        "danger": round(danger, 2),
        "warning": round(warning, 2),
    }


def load_distance_thresholds():
    if not DISTANCE_SETTINGS_FILE.is_file():
        save_distance_thresholds(
            DEFAULT_THRESHOLDS["danger"],
            DEFAULT_THRESHOLDS["warning"],
        )
        return DEFAULT_THRESHOLDS.copy()

    try:
        with DISTANCE_SETTINGS_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return DEFAULT_THRESHOLDS.copy()

    return _normalize_thresholds(data)


def save_distance_thresholds(danger, warning):
    thresholds = _normalize_thresholds({"danger": danger, "warning": warning})
    SETTINGS_DIR.mkdir(exist_ok=True)
    with DISTANCE_SETTINGS_FILE.open("w", encoding="utf-8") as file:
        json.dump(thresholds, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return thresholds
