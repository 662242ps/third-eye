"""Thai voice announcements naming the nearest DANGER-level object.

Speech clips are pre-generated Thai audio (see assets/voice/*.mp3, made
with gTTS at build time) so the app stays fully offline at runtime. They
are played through the Windows MCI API via `winmm.dll`, which can read
MP3 directly -- unlike `winsound`, which only understands WAV -- so no
extra audio library or ffmpeg install is required.
"""
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VOICE_DIR = PROJECT_ROOT / "assets" / "voice"
TTS_DIR = PROJECT_ROOT / "tts"
TTS_EXE = TTS_DIR / "thai_tts.exe"
PIPER_EXE = TTS_DIR / "piper.exe"
PIPER_MODEL = TTS_DIR / "thai_voice.onnx"
VACHANA_CONFIG = TTS_DIR / "voices" / "speaker_config.json"
TEMP_TTS_WAV = TTS_DIR / "alert_runtime.wav"
# VachanaTTS uses a larger length scale for slower speech.
SPEECH_LENGTH_SCALE = 1.3
SPEECH_VOLUME = 1.2

# Announce at most once per this many seconds for the same object/status.
# A longer cooldown prevents distance noise from becoming a stream of alerts.
COOLDOWN_S = 5.0
# On top of the cooldown, don't repeat for the same label unless its
# distance has moved at least this many meters from where it was last
# announced -- an object parked at a steady distance stays quiet instead of
# re-announcing every COOLDOWN_S.
REPEAT_DISTANCE_M = 3.0
# Keep the last alert alive briefly when a detector result disappears for a
# frame or two. This avoids treating a blinking detection as a new object.
LOST_GRACE_S = 1.5
# Require a candidate to remain visible for a short time before speaking.
STABLE_S = 0.35
MCI_ALIAS = "third_eye_voice"

THAI_LABELS = {
    "car": "\u0e23\u0e16\u0e22\u0e19\u0e15\u0e4c",
    "motorcycle": "\u0e23\u0e16\u0e08\u0e31\u0e01\u0e23\u0e22\u0e32\u0e19\u0e22\u0e19\u0e15\u0e4c",
    "truck": "\u0e23\u0e16\u0e1a\u0e23\u0e23\u0e17\u0e38\u0e01",
    "bus": "\u0e23\u0e16\u0e1a\u0e31\u0e2a",
    "person": "\u0e04\u0e19",
}

THAI_NUMBER_WORDS = {
    0: "\u0e28\u0e39\u0e19\u0e22\u0e4c",
    1: "\u0e2b\u0e19\u0e36\u0e48\u0e07", 2: "\u0e2a\u0e2d\u0e07", 3: "\u0e2a\u0e32\u0e21",
    4: "\u0e2a\u0e35\u0e48", 5: "\u0e2b\u0e49\u0e32", 6: "\u0e2b\u0e01",
    7: "\u0e40\u0e08\u0e47\u0e14", 8: "\u0e41\u0e1b\u0e14", 9: "\u0e40\u0e01\u0e49\u0e32",
    10: "\u0e2a\u0e34\u0e1a", 11: "\u0e2a\u0e34\u0e1a\u0e40\u0e2d\u0e47\u0e14", 12: "\u0e2a\u0e34\u0e1a\u0e2a\u0e2d\u0e07",
    13: "\u0e2a\u0e34\u0e1a\u0e2a\u0e32\u0e21", 14: "\u0e2a\u0e34\u0e1a\u0e2a\u0e35\u0e48", 15: "\u0e2a\u0e34\u0e1a\u0e2b\u0e49\u0e32",
    16: "\u0e2a\u0e34\u0e1a\u0e2b\u0e01", 17: "\u0e2a\u0e34\u0e1a\u0e40\u0e08\u0e47\u0e14", 18: "\u0e2a\u0e34\u0e1a\u0e41\u0e1b\u0e14",
    19: "\u0e2a\u0e34\u0e1a\u0e40\u0e01\u0e49\u0e32", 20: "\u0e22\u0e35\u0e48\u0e2a\u0e34\u0e1a", 21: "\u0e22\u0e35\u0e48\u0e2a\u0e34\u0e1a\u0e40\u0e2d\u0e47\u0e14",
    22: "\u0e22\u0e35\u0e48\u0e2a\u0e34\u0e1a\u0e2a\u0e2d\u0e07", 23: "\u0e22\u0e35\u0e48\u0e2a\u0e34\u0e1a\u0e2a\u0e32\u0e21", 24: "\u0e22\u0e35\u0e48\u0e2a\u0e34\u0e1a\u0e2a\u0e35\u0e48",
    25: "\u0e22\u0e35\u0e48\u0e2a\u0e34\u0e1a\u0e2b\u0e49\u0e32", 26: "\u0e22\u0e35\u0e48\u0e2a\u0e34\u0e1a\u0e2b\u0e01", 27: "\u0e22\u0e35\u0e48\u0e2a\u0e34\u0e1a\u0e40\u0e08\u0e47\u0e14",
    28: "\u0e22\u0e35\u0e48\u0e2a\u0e34\u0e1a\u0e41\u0e1b\u0e14", 29: "\u0e22\u0e35\u0e48\u0e2a\u0e34\u0e1a\u0e40\u0e01\u0e49\u0e32",
    30: "\u0e2a\u0e32\u0e21\u0e2a\u0e34\u0e1a", 31: "\u0e2a\u0e32\u0e21\u0e2a\u0e34\u0e1a\u0e40\u0e2d\u0e47\u0e14", 32: "\u0e2a\u0e32\u0e21\u0e2a\u0e34\u0e1a\u0e2a\u0e2d\u0e07",
    33: "\u0e2a\u0e32\u0e21\u0e2a\u0e34\u0e1a\u0e2a\u0e32\u0e21", 34: "\u0e2a\u0e32\u0e21\u0e2a\u0e34\u0e1a\u0e2a\u0e35\u0e48", 35: "\u0e2a\u0e32\u0e21\u0e2a\u0e34\u0e1a\u0e2b\u0e49\u0e32",
    36: "\u0e2a\u0e32\u0e21\u0e2a\u0e34\u0e1a\u0e2b\u0e01", 37: "\u0e2a\u0e32\u0e21\u0e2a\u0e34\u0e1a\u0e40\u0e08\u0e47\u0e14", 38: "\u0e2a\u0e32\u0e21\u0e2a\u0e34\u0e1a\u0e41\u0e1b\u0e14",
    39: "\u0e2a\u0e32\u0e21\u0e2a\u0e34\u0e1a\u0e40\u0e01\u0e49\u0e32", 40: "\u0e2a\u0e35\u0e48\u0e2a\u0e34\u0e1a",
}


def _thai_integer(value):
    """Spell distance values 0-40 from an explicit pronunciation table."""
    value = max(0, int(value))
    if value in THAI_NUMBER_WORDS:
        return THAI_NUMBER_WORDS[value]
    if 41 <= value <= 99:
        tens, ones = divmod(value, 10)
        tens_word = THAI_NUMBER_WORDS[tens] + "\u0e2a\u0e34\u0e1a"
        if ones == 0:
            return tens_word
        ones_word = "\u0e40\u0e2d\u0e47\u0e14" if ones == 1 else THAI_NUMBER_WORDS[ones]
        return tens_word + ones_word
    # Distances outside the calibrated voice table are uncommon; keep the
    # numeric fallback rather than inventing an incorrect Thai pronunciation.
    return str(value)


def _thai_distance(distance):
    # The UI keeps the precise measured value, but voice alerts use only a
    # whole meter so the announcement is shorter and easier to understand.
    # Keep the integer part shown by the UI: 15.8 -> 15 meters.
    whole = max(0, int(float(distance)))
    return _thai_integer(whole) + " \u0e40\u0e21\u0e15\u0e23"


class VoiceAnnouncer:
    """Speaks the nearest WARNING/DANGER object and its estimated distance."""

    def __init__(self):
        self.muted = False
        self.enabled = True
        self._last_label = None
        self._last_status = None
        self._last_time = 0.0
        self._last_distance = None
        self._last_seen = 0.0
        self._candidate = None
        self._candidate_since = 0.0
        self._mci = None
        self._speech_process = None
        self._has_thai_voice = None
        self._speech_lock = threading.Lock()
        self._speech_thread = None
        self._preload_thread = None
        self._voice_model = "th_f_2"
        self._vachana_model = TTS_DIR / "voices" / "th_f_2.onnx"
        self._voice_generation = 0

        if sys.platform == "win32" and VOICE_DIR.is_dir():
            try:
                import ctypes

                self._mci = ctypes.windll.winmm.mciSendStringW
            except (ImportError, OSError, AttributeError):
                self._mci = None

    @property
    def available(self):
        return self._mci is not None or self._bundled_tts_available()

    def _bundled_tts_available(self):
        return (
            (self._vachana_model.is_file() and VACHANA_CONFIG.is_file())
            or TTS_EXE.is_file()
            or (PIPER_EXE.is_file() and PIPER_MODEL.is_file())
        )

    def set_voice_model(self, model_name):
        model_name = str(model_name or "th_f_2").strip()
        if not model_name.startswith("th_"):
            model_name = "th_f_2"
        model_path = TTS_DIR / "voices" / f"{model_name}.onnx"
        if not model_path.is_file():
            model_name = "th_f_2"
            model_path = TTS_DIR / "voices" / "th_f_2.onnx"
        if model_name != self._voice_model:
            self._voice_generation += 1
            self.stop()
            self._voice_model = model_name
            self._vachana_model = model_path
            self._vachana_voice = None
            self.preload()

    @property
    def voice_model(self):
        return self._voice_model

    def set_muted(self, muted):
        self.muted = bool(muted)
        if self.muted:
            self.stop()

    def set_enabled(self, enabled):
        """Toggle this channel from the alert settings screen."""
        self.enabled = bool(enabled)
        if not self.enabled:
            self.stop()

    def announce(self, label, distance=None, status="DANGER"):
        """Call every frame for the nearest WARNING/DANGER object.

        SAFE (or no object) is deliberately silent. Short loss/stability
        windows suppress detector flicker without hiding a real transition.
        """
        now = time.monotonic()
        if label is None:
            if self._last_seen and now - self._last_seen > LOST_GRACE_S:
                self._reset_state()
            return
        self._last_seen = now
        status = status if status in ("DANGER", "WARNING") else "WARNING"
        if not self.available or self.muted or not self.enabled:
            return

        candidate = (label, status)
        if candidate != self._candidate:
            self._candidate = candidate
            self._candidate_since = now
            return
        if now - self._candidate_since < STABLE_S:
            return

        if candidate == (self._last_label, self._last_status):
            cooldown_elapsed = (now - self._last_time) >= COOLDOWN_S
            moved_away = (
                distance is None
                or self._last_distance is None
                or abs(distance - self._last_distance) >= REPEAT_DISTANCE_M
            )
            if not (cooldown_elapsed and moved_away):
                return

        self._last_label = label
        self._last_status = status
        self._last_time = now
        # `distance` is the already-smoothed detection value from
        # YoloThread (`detection["dist"]`). Speak that exact value so the
        # number matches the value shown in the UI.
        self._last_distance = distance
        self._play_message(label, status, distance)

    def _play_message(self, label, status, distance):
        # Use phonetic spelling so the bundled Thai model pronounces the
        # danger word as requested: "อันตะราย".
        thai_status = "\u0e2d\u0e31\u0e19\u0e15\u0e30\u0e23\u0e32\u0e22" if status == "DANGER" else "\u0e23\u0e30\u0e27\u0e31\u0e07"
        thai_label = THAI_LABELS.get(label, label)
        distance_text = " " + _thai_distance(distance) if distance is not None else ""
        # Commas create clear pauses around the distance without splitting
        # Thai number words (e.g. keep "สิบห้า" together).
        # Short, clearly separated phrases are easier for the Thai model to
        # pronounce than a long conversational sentence.
        # Keep the sentence short: status, object, distance. The distance
        # helper already supplies only the number and "เมตร".
        message = f"{thai_status}, มี, {thai_label}, อยู่ในระยะ, {distance_text}"

        # Prefer a self-contained offline TTS engine shipped beside the app.
        # This is the path used on other computers, so it does not depend on
        # Windows having a Thai speech voice installed.
        if self._play_bundled_tts(message):
            return

        # SAPI can synthesize the changing distance at runtime. It avoids
        # needing a separate audio file for every possible meter value.
        if sys.platform == "win32" and self._thai_voice_installed():
            try:
                if self._speech_process and self._speech_process.poll() is None:
                    self._speech_process.terminate()
                import base64

                encoded_message = base64.b64encode(
                    message.encode("utf-16-le")
                ).decode("ascii")
                script = (
                    "Add-Type -AssemblyName System.Speech; "
                    "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    "$v=$s.GetInstalledVoices() | Where-Object "
                    "{$_.VoiceInfo.Culture.TwoLetterISOLanguageName -eq 'th'} | Select-Object -First 1; "
                    "if (-not $v) {exit 2}; "
                    "$s.SelectVoice($v.VoiceInfo.Name); "
                    "$m=[Text.Encoding]::Unicode.GetString([Convert]::FromBase64String('"
                    + encoded_message
                    + "')); $s.Speak($m)"
                )
                self._speech_process = subprocess.Popen(
                    ["powershell", "-NoProfile", "-Command", script],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return
            except (OSError, BrokenPipeError):
                if self._speech_process and self._speech_process.poll() is None:
                    self._speech_process.terminate()

        # Legacy clips do not contain a distance. Only use them when there is
        # no measured distance (for example, a manually-triggered generic alert).
        if distance is not None:
            return
        path = VOICE_DIR / f"{label}_danger.mp3"
        if not path.is_file():
            path = VOICE_DIR / "generic_danger.mp3"
        if path.is_file():
            self._play(path)

    def _play_bundled_tts(self, message):
        if sys.platform != "win32":
            return False
        if not self._bundled_tts_available():
            return False

        # One utterance at a time. Generating/playing another WAV over the
        # previous one was the reason speech was being cut off.
        if self._speech_lock.locked():
            return True
        self._speech_thread = threading.Thread(
            target=self._synthesize_and_play, args=(message,), daemon=True
        )
        self._speech_thread.start()
        return True

    def preload(self):
        """Load the bundled voice before the first alert is needed."""
        if sys.platform != "win32" or not self._vachana_model.is_file():
            return
        if (
            getattr(self, "_vachana_voice", None) is not None
            or self._speech_lock.locked()
            or (self._preload_thread and self._preload_thread.is_alive())
        ):
            return
        self._preload_thread = threading.Thread(
            target=self._load_bundled_voice, daemon=True
        )
        self._preload_thread.start()

    def _load_bundled_voice(self):
        generation = self._voice_generation
        try:
            from vachanatts.voice import Voice

            model_path = self._vachana_model
            loaded_voice = Voice.load(model_path, VACHANA_CONFIG)
            with self._speech_lock:
                if generation == self._voice_generation:
                    self._vachana_voice = loaded_voice
        except (ImportError, OSError, RuntimeError):
            pass
        finally:
            self._preload_thread = None
            # If the user changed models while this load was in progress,
            # queue the newly selected model without blocking the UI thread.
            if generation != self._voice_generation:
                self.preload()

    def _synthesize_and_play(self, message):
        output_path = TTS_DIR / f"alert_{time.time_ns()}.wav"
        generation = self._voice_generation
        model_path = self._vachana_model
        with self._speech_lock:
            try:
                if model_path.is_file() and VACHANA_CONFIG.is_file():
                    import wave
                    from vachanatts.config import SpeechConfig
                    from vachanatts.voice import Voice

                    if getattr(self, "_vachana_voice", None) is None:
                        self._vachana_voice = Voice.load(model_path, VACHANA_CONFIG)
                    with wave.open(str(output_path), "wb") as wav_file:
                        self._vachana_voice.synthesize_wav(
                            message,
                            wav_file,
                            SpeechConfig(
                                volume=SPEECH_VOLUME,
                                length_scale=SPEECH_LENGTH_SCALE,
                            ),
                        )
                elif TTS_EXE.is_file():
                    subprocess.run(
                        [str(TTS_EXE), "--text", message, "--output", str(output_path)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=8,
                        check=True,
                    )
                elif PIPER_EXE.is_file() and PIPER_MODEL.is_file():
                    subprocess.run(
                        [str(PIPER_EXE), "--model", str(PIPER_MODEL),
                         "--output_file", str(output_path)],
                        input=message, text=True,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        timeout=8, check=True,
                    )

                if output_path.is_file() and generation == self._voice_generation:
                    import winsound

                    # Synchronous playback in this worker prevents the file
                    # from being replaced/deleted while Windows is reading it.
                    winsound.PlaySound(str(output_path), winsound.SND_FILENAME)
            except (OSError, subprocess.SubprocessError):
                pass
            finally:
                try:
                    output_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _thai_voice_installed(self):
        """Return false instead of silently letting SAPI use English."""
        if self._has_thai_voice is not None:
            return self._has_thai_voice
        try:
            check = subprocess.run(
                [
                    "powershell", "-NoProfile", "-Command",
                    "Add-Type -AssemblyName System.Speech; "
                    "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    "if ($s.GetInstalledVoices() | Where-Object "
                    "{$_.VoiceInfo.Culture.TwoLetterISOLanguageName -eq 'th'}) {exit 0} else {exit 1}",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
            self._has_thai_voice = check.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            self._has_thai_voice = False
        return self._has_thai_voice

    def _play(self, path):
        buf_size = 256
        import ctypes

        buf = ctypes.create_unicode_buffer(buf_size)
        # Close any clip still playing so the newest object always wins.
        self._mci(f"close {MCI_ALIAS}", buf, buf_size - 1, 0)
        self._mci(f'open "{path}" type mpegvideo alias {MCI_ALIAS}', buf, buf_size - 1, 0)
        self._mci(f"play {MCI_ALIAS}", buf, buf_size - 1, 0)

    def stop(self):
        self._reset_state()
        if sys.platform == "win32":
            try:
                import winsound

                winsound.PlaySound(None, 0)
            except (ImportError, RuntimeError):
                pass
        if self._speech_process and self._speech_process.poll() is None:
            self._speech_process.terminate()
        if self._mci is not None:
            buf = None
            import ctypes

            buf = ctypes.create_unicode_buffer(256)
            self._mci(f"close {MCI_ALIAS}", buf, 255, 0)

    def _reset_state(self):
        self._last_label = None
        self._last_status = None
        self._last_distance = None
        self._last_seen = 0.0
        self._candidate = None
        self._candidate_since = 0.0
