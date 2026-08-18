"""Generate all offline Thai alert-word WAV files once.

Run from the project root with the project's virtual environment active:

    python tools/generate_voice_segments.py

The bundled ONNX model is used only during this one-time generation step.
The application then concatenates the generated WAV files without loading
VachanaTTS during live video.
"""

import argparse
import os
import sys
import wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from vachanatts.config import SpeechConfig

from vision.voice_alert import _load_vachana_voice
from vision.voice_segments import (
    SEGMENT_LENGTH_SCALE,
    SEGMENT_LENGTH_SCALES,
    SEGMENT_TEXT,
    SEGMENT_TTS_TEXT,
    trim_wav_file,
)


VOICE_DIR = PROJECT_ROOT / "tts" / "voices"
OUTPUT_DIR = PROJECT_ROOT / "assets" / "voice" / "segments"


def generate(voice_model=None, force=False):
    if voice_model:
        model_paths = [VOICE_DIR / f"{voice_model}.onnx"]
    else:
        model_paths = sorted(VOICE_DIR.glob("th_*.onnx"))

    generated = 0
    skipped = 0

    for model_path in model_paths:
        if not model_path.is_file():
            print(f"missing voice model: {model_path}")
            continue
        voice = _load_vachana_voice(model_path)
        model_output_dir = OUTPUT_DIR / model_path.stem
        model_output_dir.mkdir(parents=True, exist_ok=True)
        for key, text in SEGMENT_TEXT.items():
            output_path = model_output_dir / f"{key}.wav"
            if output_path.is_file() and output_path.stat().st_size > 44 and not force:
                skipped += 1
                continue

            temporary_path = output_path.with_suffix(".tmp.wav")
            try:
                with wave.open(str(temporary_path), "wb") as wav_file:
                    voice.synthesize_wav(
                        SEGMENT_TTS_TEXT.get(key, text),
                        wav_file,
                        SpeechConfig(
                            volume=1.0,
                            length_scale=SEGMENT_LENGTH_SCALES.get(
                                key, SEGMENT_LENGTH_SCALE
                            ),
                        ),
                    )
                trim_wav_file(temporary_path)
                os.replace(temporary_path, output_path)
                generated += 1
                print(f"generated {model_path.stem}/{key}.wav")
            finally:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    print(
        f"completed: generated={generated}, skipped={skipped}, "
        f"total_per_voice={len(SEGMENT_TEXT)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--voice-model",
        help="generate one model only, e.g. th_m_1 (default: every th_*.onnx model)",
    )
    parser.add_argument("--force", action="store_true", help="regenerate existing WAV files")
    args = parser.parse_args()
    generate(voice_model=args.voice_model, force=args.force)
