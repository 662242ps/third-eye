"""Thai voice announcements naming the nearest DANGER-level object.

Speech clips are pre-generated Thai audio (see assets/voice/*.mp3, made
with gTTS at build time) so the app stays fully offline at runtime. They
are played through the Windows MCI API via `winmm.dll`, which can read
MP3 directly -- unlike `winsound`, which only understands WAV -- so no
extra audio library or ffmpeg install is required.
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VOICE_DIR = PROJECT_ROOT / "assets" / "voice"

# Announce at most once per this many seconds for the *same* label, so a
# lingering danger object doesn't get repeated every frame.
COOLDOWN_S = 3.0
# On top of the cooldown, don't repeat for the same label unless its
# distance has moved at least this many meters from where it was last
# announced -- an object parked at a steady distance stays quiet instead of
# re-announcing every COOLDOWN_S.
REPEAT_DISTANCE_M = 3.0
MCI_ALIAS = "third_eye_voice"


class VoiceAnnouncer:
    """Speaks the label of the nearest object currently in DANGER range."""

    def __init__(self):
        self.muted = False
        self.enabled = True
        self._last_label = None
        self._last_time = 0.0
        self._last_distance = None
        self._mci = None

        if sys.platform == "win32" and VOICE_DIR.is_dir():
            try:
                import ctypes

                self._mci = ctypes.windll.winmm.mciSendStringW
            except (ImportError, OSError, AttributeError):
                self._mci = None

    @property
    def available(self):
        return self._mci is not None

    def set_muted(self, muted):
        self.muted = bool(muted)

    def set_enabled(self, enabled):
        """Toggle this channel from the alert settings screen."""
        self.enabled = bool(enabled)

    def announce(self, label, distance=None):
        """Call every frame. `label` is the nearest DANGER object's class
        name (or None when nothing is currently in danger range), and
        `distance` is its distance in meters."""
        if label is None:
            self._last_label = None
            self._last_distance = None
            return
        if not self.available or self.muted or not self.enabled:
            return

        now = time.monotonic()
        if label == self._last_label:
            cooldown_elapsed = (now - self._last_time) >= COOLDOWN_S
            moved_away = (
                distance is None
                or self._last_distance is None
                or abs(distance - self._last_distance) >= REPEAT_DISTANCE_M
            )
            if not (cooldown_elapsed and moved_away):
                return

        path = VOICE_DIR / f"{label}_danger.mp3"
        if not path.is_file():
            path = VOICE_DIR / "generic_danger.mp3"
            if not path.is_file():
                return

        self._last_label = label
        self._last_time = now
        self._last_distance = distance
        self._play(path)

    def _play(self, path):
        buf_size = 256
        import ctypes

        buf = ctypes.create_unicode_buffer(buf_size)
        # Close any clip still playing so the newest object always wins.
        self._mci(f"close {MCI_ALIAS}", buf, buf_size - 1, 0)
        self._mci(f'open "{path}" type mpegvideo alias {MCI_ALIAS}', buf, buf_size - 1, 0)
        self._mci(f"play {MCI_ALIAS}", buf, buf_size - 1, 0)

    def stop(self):
        self._last_label = None
        if self.available:
            buf = None
            import ctypes

            buf = ctypes.create_unicode_buffer(256)
            self._mci(f"close {MCI_ALIAS}", buf, 255, 0)
