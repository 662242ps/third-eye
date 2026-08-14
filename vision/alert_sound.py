"""Two-level looping siren for WARNING and DANGER detections.

The application targets Windows, so the standard-library ``winsound``
module is used.  The short WAV patterns are generated with the Python
standard library so that each level can have its own frequency, loudness,
and pause between repetitions without adding an audio dependency.
"""

import math
import struct
import sys
import tempfile
import wave
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALERT_SOUND_FILE = PROJECT_ROOT / "assets" / "alert_danger.wav"

# Values are deliberately separated here so the difference between levels is
# easy to tune and test.  ``volume`` is the PCM amplitude, not the Windows
# master-volume setting, so changing the warning alarm cannot change other
# application sounds.
ALARM_PROFILES = {
    "WARNING": {
        "frequency_hz": (650, 800),
        "volume": 0.28,
        "tone_ms": 220,
        "pause_ms": 1050,
        "step_ms": 110,
    },
    "DANGER": {
        "frequency_hz": (950, 1250),
        "volume": 0.82,
        "tone_ms": 480,
        "pause_ms": 150,
        "step_ms": 120,
    },
}


def _write_siren_wav(path, profile):
    """Create one looping siren cycle if it is not already present."""
    if path.is_file() and path.stat().st_size > 44:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 44100
    tone_seconds = profile["tone_ms"] / 1000.0
    pause_seconds = profile["pause_ms"] / 1000.0
    step_seconds = profile["step_ms"] / 1000.0
    cycle_seconds = tone_seconds + pause_seconds
    low_hz, high_hz = profile["frequency_hz"]
    amplitude = int(32767 * profile["volume"])
    frame_count = int(sample_rate * cycle_seconds)
    step_frames = max(1, int(sample_rate * step_seconds))
    fade_frames = max(1, int(sample_rate * 0.012))

    frames = bytearray()
    for frame in range(frame_count):
        cycle_position = frame / sample_rate
        if cycle_position >= tone_seconds:
            sample = 0
        else:
            step_position = frame % step_frames
            frequency = high_hz if (frame // step_frames) % 2 else low_hz
            phase = 2.0 * math.pi * frequency * step_position / sample_rate
            envelope = min(
                1.0,
                step_position / fade_frames,
                (step_frames - step_position) / fade_frames,
            )
            sample = int(amplitude * envelope * math.sin(phase))
        frames.extend(struct.pack("<h", sample))

    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(frames)
    return path


class DangerAlarm:
    """Start/stop a level-aware siren, muting itself when unavailable."""

    def __init__(self):
        self.muted = False
        self.enabled = True
        self.volume = 100
        self._active = False
        self._level = "DANGER"
        self._winsound = None
        self._sound_files = {}

        if sys.platform == "win32":
            try:
                import winsound

                self._winsound = winsound
                # Keep generated files in the OS temp directory rather than
                # changing the repository or relying on a writable install
                # directory.  They are reused on subsequent app launches.
                sound_dir = Path(tempfile.gettempdir()) / "third_eye_siren"
                self._sound_dir = sound_dir
                self._rebuild_sound_files()
            except ImportError:
                self._winsound = None

    @property
    def available(self):
        return self._winsound is not None and bool(self._sound_files)

    def set_muted(self, muted):
        self.muted = bool(muted)
        if self.muted and self._active:
            self._active = False
            self._play(False)

    def set_enabled(self, enabled):
        """Toggle this channel from the alert settings screen."""
        self.enabled = bool(enabled)
        if not self.enabled and self._active:
            self._active = False
            self._play(False)

    def set_volume(self, volume):
        """Set siren loudness without changing warning/danger profiles."""
        try:
            volume = int(round(float(volume)))
        except (TypeError, ValueError):
            volume = 100
        self.volume = max(0, min(100, volume))
        if self._winsound is None:
            return
        was_active = self._active
        if was_active:
            self._active = False
            self._play(False)
        self._rebuild_sound_files()
        if was_active and self.volume > 0:
            self._active = True
            self._play(True)

    def _rebuild_sound_files(self):
        self._sound_files = {}
        sound_dir = getattr(self, "_sound_dir", None)
        if sound_dir is None:
            return
        for level, profile in ALARM_PROFILES.items():
            scaled_profile = dict(profile)
            scaled_profile["volume"] = profile["volume"] * self.volume / 100.0
            sound_path = sound_dir / f"siren_{level.lower()}_v2_{self.volume}.wav"
            try:
                self._sound_files[level] = _write_siren_wav(
                    sound_path, scaled_profile
                )
            except (OSError, ValueError, wave.Error):
                # A read-only temp directory should not prevent the
                # application from starting. Retain the original danger
                # sound as a safe fallback when available.
                if level == "DANGER" and ALERT_SOUND_FILE.is_file():
                    self._sound_files[level] = ALERT_SOUND_FILE

    def set_active(self, active, level="DANGER"):
        """Play the appropriate siren for the nearest WARNING/DANGER object."""
        level = str(level).upper()
        if level not in ALARM_PROFILES:
            level = "DANGER"
        active = (
            bool(active)
            and self.volume > 0
            and not self.muted
            and self.enabled
        )
        if active and level not in self._sound_files:
            active = False

        level_changed = level != self._level
        if active == self._active and not level_changed:
            return
        self._active = active
        self._level = level
        self._play(active)

    def _play(self, active):
        if self._winsound is None:
            return
        if active:
            sound_file = self._sound_files.get(self._level)
            if sound_file is None:
                return
            self._winsound.PlaySound(
                str(sound_file),
                self._winsound.SND_FILENAME
                | self._winsound.SND_ASYNC
                | self._winsound.SND_LOOP,
            )
        else:
            self._winsound.PlaySound(None, 0)

    def stop(self):
        self._active = False
        self._play(False)
