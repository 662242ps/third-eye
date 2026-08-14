"""Persisted on/off state for the siren and voice alert channels."""
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_DIR = PROJECT_ROOT / "settings"
ALERT_SETTINGS_FILE = SETTINGS_DIR / "alert_settings.json"

DEFAULT_ALERT_SETTINGS = {
    "siren_enabled": True,
    "voice_enabled": True,
    "voice_model": "th_m_1",
    "siren_volume": 100,
    "voice_volume": 100,
}


def _as_bool(value, default):
    """Normalize JSON booleans without treating the string 'false' as true."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "on"}:
            return True
        if normalized in {"false", "no", "0", "off"}:
            return False
    return default


def _as_volume(value, default=100):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return max(0, min(100, int(round(value))))


def _normalize(data):
    try:
        siren_enabled = _as_bool(data.get("siren_enabled", True), True)
        voice_enabled = _as_bool(data.get("voice_enabled", True), True)
        voice_model = str(data.get("voice_model", "th_m_1")).strip() or "th_m_1"
        siren_volume = _as_volume(data.get("siren_volume", 100))
        voice_volume = _as_volume(data.get("voice_volume", 100))
    except AttributeError:
        return DEFAULT_ALERT_SETTINGS.copy()
    if voice_model != "th_m_1":
        voice_model = DEFAULT_ALERT_SETTINGS["voice_model"]
    return {
        "siren_enabled": siren_enabled,
        "voice_enabled": voice_enabled,
        "voice_model": voice_model,
        "siren_volume": siren_volume,
        "voice_volume": voice_volume,
    }


def load_alert_settings():
    if not ALERT_SETTINGS_FILE.is_file():
        return DEFAULT_ALERT_SETTINGS.copy()
    try:
        with ALERT_SETTINGS_FILE.open("r", encoding="utf-8") as file:
            return _normalize(json.load(file))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_ALERT_SETTINGS.copy()


def save_alert_settings(
    siren_enabled,
    voice_enabled,
    voice_model="th_m_1",
    siren_volume=100,
    voice_volume=100,
):
    settings = _normalize(
        {
            "siren_enabled": siren_enabled,
            "voice_enabled": voice_enabled,
            "voice_model": voice_model,
            "siren_volume": siren_volume,
            "voice_volume": voice_volume,
        }
    )
    SETTINGS_DIR.mkdir(exist_ok=True)
    with ALERT_SETTINGS_FILE.open("w", encoding="utf-8") as file:
        json.dump(settings, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return settings
