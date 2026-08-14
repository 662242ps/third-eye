"""Thai voice segment catalog used by the real-time alert path.

The segments are generated once from the bundled Vachana voice and then
concatenated as PCM WAV files at runtime. This keeps ONNX out of the live
video loop while retaining dynamic object names and integer distances.
"""

import os
import wave
from pathlib import Path


SEGMENT_TEXT = {
    "danger": "\u0e2d\u0e31\u0e19\u0e15\u0e23\u0e32\u0e22",
    "warning": "\u0e23\u0e30\u0e27\u0e31\u0e07",
    "has": "\u0e21\u0e35",
    "in_range": "\u0e2d\u0e22\u0e39\u0e48\u0e43\u0e19\u0e23\u0e30\u0e22\u0e30",
    # Keep the requested phrase text for the alert catalog.
    "meter": "\u0e40\u0e21\u0e15\u0e23   ",
    "car": "\u0e23\u0e16\u0e22\u0e19\u0e15\u0e4c",
    "motorcycle": "\u0e23\u0e16\u0e08\u0e31\u0e01\u0e23\u0e22\u0e32\u0e19\u0e22\u0e19\u0e15\u0e4c",
    "truck": "\u0e23\u0e16\u0e1a\u0e23\u0e23\u0e17\u0e38\u0e01",
    "bus": "\u0e23\u0e16\u0e1a\u0e31\u0e2a",
    "person": "\u0e04\u0e19",
}

OBJECT_SEGMENTS = {
    "car": "car",
    "motorcycle": "motorcycle",
    "truck": "truck",
    "bus": "bus",
    "person": "person",
}

_DIGITS = {
    0: "\u0e28\u0e39\u0e19\u0e22\u0e4c",
    1: "\u0e2b\u0e19\u0e36\u0e48\u0e07",
    2: "\u0e2a\u0e2d\u0e07",
    3: "\u0e2a\u0e32\u0e21",
    4: "\u0e2a\u0e35\u0e48",
    5: "\u0e2b\u0e49\u0e32",
    6: "\u0e2b\u0e01",
    7: "\u0e40\u0e08\u0e47\u0e14",
    8: "\u0e41\u0e1b\u0e14",
    9: "\u0e40\u0e01\u0e49\u0e32",
}

# Smaller values make the Vachana voice speak faster. The generated clips use
# this value, while the runtime also trims the model's large leading/trailing
# silence padding before joining clips.
SEGMENT_LENGTH_SCALE = 0.75
# Very short unit words need their natural duration. Keeping "เมตร" at the
# fast global scale makes its final consonant sound clipped.
SEGMENT_LENGTH_SCALES = {
    "meter": 1.0,
}
# Vachana drops the final consonant when synthesizing the isolated word
# "เมตร". Use its natural Thai pronunciation only for the generated WAV;
# the catalog/text shown by the application remains "เมตร   ".
SEGMENT_TTS_TEXT = {
    "meter": "\u0e40\u0e21\u0e49\u0e15",
}
# Keep quiet consonants and the end of short words such as "เมตร". The old
# value (256) removed the low-volume tail of some Thai syllables.
SEGMENT_TRIM_THRESHOLD = 64


def thai_integer(value):
    value = max(0, min(99, int(value)))
    if value < 10:
        return _DIGITS[value]
    if value < 20:
        ones = value - 10
        return "\u0e2a\u0e34\u0e1a" + ("\u0e40\u0e2d\u0e47\u0e14" if ones == 1 else _DIGITS[ones])
    tens, ones = divmod(value, 10)
    tens_word = "\u0e22\u0e35\u0e48" if tens == 2 else _DIGITS[tens]
    return tens_word + "\u0e2a\u0e34\u0e1a" + (
        "" if ones == 0 else "\u0e40\u0e2d\u0e47\u0e14" if ones == 1 else _DIGITS[ones]
    )


for _number in range(100):
    SEGMENT_TEXT[f"number_{_number:02d}"] = thai_integer(_number)


def build_segment_keys(label, status, distance):
    """Return the exact WAV segment sequence for one alert."""
    status_key = "danger" if status == "DANGER" else "warning"
    object_key = OBJECT_SEGMENTS.get(str(label).lower())
    if object_key is None:
        return []
    try:
        number = max(0, min(99, int(float(distance))))
    except (TypeError, ValueError):
        number = 0
    return [
        status_key,
        "has",
        object_key,
        "in_range",
        f"number_{number:02d}",
        "meter",
    ]


def segment_paths(segment_dir, label, status, distance):
    keys = build_segment_keys(label, status, distance)
    if not keys:
        return []
    root = Path(segment_dir)
    paths = [root / f"{key}.wav" for key in keys]
    return paths if all(path.is_file() for path in paths) else []


def available_voice_models(segment_root):
    """Return voice names whose complete WAV catalog is ready for playback."""
    root = Path(segment_root)
    if not root.is_dir():
        return []
    required = tuple(SEGMENT_TEXT)
    models = []
    try:
        candidates = sorted(path for path in root.iterdir() if path.is_dir())
    except OSError:
        return []
    for candidate in candidates:
        try:
            complete = all((candidate / f"{key}.wav").is_file() for key in required)
        except OSError:
            complete = False
        if complete:
            models.append(candidate.name)
    return models


def trim_pcm_frames(
    data,
    sample_width,
    channels,
    threshold=SEGMENT_TRIM_THRESHOLD,
    padding_frames=0,
):
    """Remove quiet PCM frames at both ends of a generated voice clip.

    VachanaTTS adds generous padding around short words. Keeping that padding
    on every segment makes a six-part alert sound unnaturally slow. This
    helper deliberately avoids ``audioop`` because that module is deprecated
    in newer Python versions.
    """
    if not data or sample_width <= 0 or channels <= 0:
        return data
    frame_bytes = sample_width * channels
    usable_length = len(data) - (len(data) % frame_bytes)
    if usable_length <= 0:
        return b""

    def frame_is_audible(offset):
        for channel in range(channels):
            sample_offset = offset + channel * sample_width
            raw = data[sample_offset:sample_offset + sample_width]
            if sample_width == 1:
                level = abs(raw[0] - 128)
            else:
                level = abs(int.from_bytes(raw, "little", signed=True))
            if level >= threshold:
                return True
        return False

    first = None
    for offset in range(0, usable_length, frame_bytes):
        if frame_is_audible(offset):
            first = offset
            break
    if first is None:
        return b""

    last = usable_length
    for offset in range(usable_length - frame_bytes, first - frame_bytes, -frame_bytes):
        if frame_is_audible(offset):
            last = offset + frame_bytes
            break
    try:
        padding_frames = max(0, int(padding_frames))
    except (TypeError, ValueError):
        padding_frames = 0
    padding_bytes = padding_frames * frame_bytes
    first = max(0, first - padding_bytes)
    last = min(usable_length, last + padding_bytes)
    return data[first:last]


def trim_wav_file(path):
    """Trim a WAV file in place, preserving its format metadata."""
    path = Path(path)
    with wave.open(str(path), "rb") as source:
        params = source.getparams()
        data = source.readframes(source.getnframes())
    trimmed = trim_pcm_frames(data, params.sampwidth, params.nchannels)
    if not trimmed or trimmed == data:
        return

    temporary_path = path.with_name(path.name + ".trim")
    try:
        with wave.open(str(temporary_path), "wb") as target:
            target.setparams(params)
            target.writeframes(trimmed)
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
