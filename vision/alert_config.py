"""Persisted on/off state for the two DANGER alert channels (siren + voice)."""
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_DIR = PROJECT_ROOT / "settings"
ALERT_SETTINGS_FILE = SETTINGS_DIR / "alert_settings.json"

DEFAULT_ALERT_SETTINGS = {
    "siren_enabled": True,
    "voice_enabled": True,
}


def _normalize(data):
    try:
        siren_enabled = bool(data.get("siren_enabled", True))
        voice_enabled = bool(data.get("voice_enabled", True))
    except AttributeError:
        return DEFAULT_ALERT_SETTINGS.copy()
    return {"siren_enabled": siren_enabled, "voice_enabled": voice_enabled}


def load_alert_settings():
    if not ALERT_SETTINGS_FILE.is_file():
        return DEFAULT_ALERT_SETTINGS.copy()
    try:
        with ALERT_SETTINGS_FILE.open("r", encoding="utf-8") as file:
            return _normalize(json.load(file))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_ALERT_SETTINGS.copy()


def save_alert_settings(siren_enabled, voice_enabled):
    settings = _normalize(
        {"siren_enabled": siren_enabled, "voice_enabled": voice_enabled}
    )
    SETTINGS_DIR.mkdir(exist_ok=True)
    with ALERT_SETTINGS_FILE.open("w", encoding="utf-8") as file:
        json.dump(settings, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return settings
