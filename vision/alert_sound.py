"""Looping audio alarm for DANGER-level detections.

Uses the stdlib `winsound` module (Windows only, matches this app's target
platform) so no extra multimedia dependency is required.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALERT_SOUND_FILE = PROJECT_ROOT / "assets" / "alert_danger.wav"


class DangerAlarm:
    """Starts/stops a looping alarm sound, muting itself when unavailable."""

    def __init__(self):
        self.muted = False
        self.enabled = True
        self._active = False
        self._winsound = None

        if sys.platform == "win32" and ALERT_SOUND_FILE.is_file():
            try:
                import winsound

                self._winsound = winsound
            except ImportError:
                self._winsound = None

    @property
    def available(self):
        return self._winsound is not None

    def set_muted(self, muted):
        self.muted = bool(muted)
        if self.muted and self._active:
            self._play(False)

    def set_enabled(self, enabled):
        """Toggle this channel from the alert settings screen."""
        self.enabled = bool(enabled)
        if not self.enabled and self._active:
            self._active = False
            self._play(False)

    def set_active(self, active):
        """Call every frame with whether any tracked object is in DANGER."""
        active = bool(active) and not self.muted and self.enabled
        if active == self._active:
            return
        self._active = active
        self._play(active)

    def _play(self, active):
        if self._winsound is None:
            return
        if active:
            self._winsound.PlaySound(
                str(ALERT_SOUND_FILE),
                self._winsound.SND_FILENAME
                | self._winsound.SND_ASYNC
                | self._winsound.SND_LOOP,
            )
        else:
            self._winsound.PlaySound(None, 0)

    def stop(self):
        self._active = False
        self._play(False)
