"""Selectable YOLO model weights.

Any ``*.pt`` file placed anywhere under ``models/`` becomes selectable from
the UI -- the file no longer has to be named ``best.pt``. Files can also be
imported (copied in) from anywhere on disk via :func:`import_model_file`.
"""
import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
SETTINGS_DIR = PROJECT_ROOT / "settings"
MODEL_SETTINGS_FILE = SETTINGS_DIR / "model_settings.json"

DEFAULT_MODEL_RELATIVE = "best.pt"


def list_available_models():
    """Return every ``*.pt`` under ``models/`` as ``{"path", "label"}`` dicts.

    ``path`` is POSIX-style and relative to ``models/`` (stable to store in
    JSON and to hand back to :func:`resolve_model_path`). ``label`` is a
    human-friendly name: the file name itself, or ``<folder>`` when the file
    is generically named ``best.pt`` inside a subfolder (e.g. training runs
    saved as ``models/model6_300/best.pt``).
    """
    if not MODELS_DIR.is_dir():
        return []

    models = []
    for pt_file in MODELS_DIR.rglob("*.pt"):
        if not pt_file.is_file():
            continue
        relative = pt_file.relative_to(MODELS_DIR)
        if relative.parent == Path("."):
            label = pt_file.stem
        elif pt_file.stem.lower() == "best":
            label = relative.parent.as_posix()
        else:
            label = f"{relative.parent.as_posix()}/{pt_file.stem}"
        models.append({"path": relative.as_posix(), "label": label})

    models.sort(key=lambda item: item["label"].lower())
    return models


def resolve_model_path(relative_path):
    """Resolve a stored relative path back to an absolute file under models/."""
    return (MODELS_DIR / relative_path).resolve()


def _is_inside_models_dir(candidate):
    try:
        candidate.relative_to(MODELS_DIR.resolve())
        return True
    except ValueError:
        return False


def load_model_settings():
    """Return ``(relative_path, absolute_path)`` for the selected model.

    Falls back to ``models/best.pt`` and then to the first available model
    if the saved selection is missing or no longer exists on disk.
    """
    selected_relative = None
    if MODEL_SETTINGS_FILE.is_file():
        try:
            with MODEL_SETTINGS_FILE.open("r", encoding="utf-8") as file:
                data = json.load(file)
            raw = str(data.get("model_path", "")).strip()
            selected_relative = raw or None
        except (OSError, json.JSONDecodeError, AttributeError):
            selected_relative = None

    if selected_relative:
        candidate = resolve_model_path(selected_relative)
        if _is_inside_models_dir(candidate) and candidate.is_file():
            return selected_relative, candidate

    default_path = MODELS_DIR / DEFAULT_MODEL_RELATIVE
    if default_path.is_file():
        return DEFAULT_MODEL_RELATIVE, default_path

    available = list_available_models()
    if available:
        return available[0]["path"], resolve_model_path(available[0]["path"])

    return DEFAULT_MODEL_RELATIVE, default_path


def save_model_settings(relative_path):
    SETTINGS_DIR.mkdir(exist_ok=True)
    with MODEL_SETTINGS_FILE.open("w", encoding="utf-8") as file:
        json.dump({"model_path": relative_path}, file, ensure_ascii=False, indent=2)
        file.write("\n")


def import_model_file(source_path):
    """Copy an external ``.pt`` file into ``models/`` and return its new
    relative path. Renames on collision instead of overwriting."""
    source = Path(source_path)
    if not source.is_file() or source.suffix.lower() != ".pt":
        raise ValueError("กรุณาเลือกไฟล์โมเดล .pt")

    MODELS_DIR.mkdir(exist_ok=True)
    destination = MODELS_DIR / source.name
    counter = 1
    while destination.exists():
        destination = MODELS_DIR / f"{source.stem}_{counter}{source.suffix}"
        counter += 1

    shutil.copy2(source, destination)
    return destination.relative_to(MODELS_DIR).as_posix()
