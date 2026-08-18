"""Thai voice announcements for the nearest WARNING/DANGER object.

The live path joins pre-generated WAV segments from
``assets/voice/segments/`` and plays the result asynchronously. This keeps
ONNX/TTS synthesis out of the video loop. The older bundled-TTS and MP3
paths remain as compatibility fallbacks when an installation is incomplete
or the segment-only flag is deliberately disabled.
"""
import json
import hashlib
import os
import subprocess
import sys
import tempfile
import threading
import time
import wave
from importlib.util import find_spec
from pathlib import Path

from vision.voice_segments import segment_paths, trim_pcm_frames

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VOICE_DIR = PROJECT_ROOT / "assets" / "voice"
SEGMENT_VOICE_DIR = VOICE_DIR / "segments"
TTS_DIR = PROJECT_ROOT / "tts"
TTS_EXE = TTS_DIR / "thai_tts.exe"
PIPER_EXE = TTS_DIR / "piper.exe"
PIPER_MODEL = TTS_DIR / "thai_voice.onnx"
VACHANA_CONFIG = TTS_DIR / "voices" / "speaker_config.json"
TEMP_TTS_DIR = Path(tempfile.gettempdir()) / "third_eye_tts"
# Keep the utterance at the model's normal speed. A slower scale creates more
# audio samples and consumes more CPU for no alerting benefit.
SPEECH_LENGTH_SCALE = 1.0
SPEECH_VOLUME = 1.2
# Keep the offline ONNX voice from creating a large worker pool. Voice
# synthesis is optional; it must never be allowed to starve the video and
# detector threads on CPU-only machines.
TTS_INTRA_OP_THREADS = 1
TTS_INTER_OP_THREADS = 1
TTS_THREAD_PRIORITY = -2  # Windows THREAD_PRIORITY_LOWEST
TTS_CACHE_LIMIT = 8
# Runtime alerts can produce a different combined WAV for each object/status
# and integer distance. Keep the temporary cache bounded across app launches
# so changing volume or testing many videos cannot grow it forever.
SEGMENT_CACHE_MAX_FILES = 128
VOLUME_CACHE_MAX_FILES = 32
SEGMENT_GAP_MS = 70
SEGMENT_FINAL_PADDING_MS = 120
# Bump this whenever the generated segment speed/padding policy changes so an
# old slow combined WAV in %TEMP% cannot be reused by the new code.
SEGMENT_CACHE_VERSION = "male-only-v11"
# Speech is enabled again, but runtime playback remains WAV-only. The Vachana
# model is used only by tools/generate_voice_segments.py during asset creation;
# live detection never loads or synthesizes with that model.
VOICE_SPEECH_ENABLED = True
SEGMENT_VOICE_ONLY = True

# Announce at most once per this many seconds for the same object/status.
# A longer cooldown prevents distance noise from becoming a stream of alerts.
COOLDOWN_S = 8.0
# On top of the cooldown, don't repeat for the same label unless its
# distance has moved at least this many meters from where it was last
# announced -- an object parked at a steady distance stays quiet instead of
# re-announcing every COOLDOWN_S.
REPEAT_DISTANCE_M = 5.0
# Keep the last alert alive briefly when a detector result disappears for a
# frame or two. This avoids treating a blinking detection as a new object.
LOST_GRACE_S = 1.5
# Require a candidate to remain visible for a short time before speaking.
STABLE_S = 0.35
MCI_ALIAS = "third_eye_voice"


def _prune_wav_cache(directory, max_files):
    """Keep only the newest generated WAV files in a temporary cache."""
    try:
        directory = Path(directory)
        files = [
            path for path in directory.glob("*.wav")
            if path.is_file()
        ]
        files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        for stale in files[max(1, int(max_files)) :]:
            try:
                stale.unlink()
            except OSError:
                # Windows may briefly keep a file open while winsound is
                # finishing playback. It can be removed on a later pass.
                continue
    except (OSError, TypeError, ValueError):
        return

THAI_LABELS = {
    "car": "\u0e23\u0e16\u0e22\u0e19\u0e15\u0e4c",
    "motorcycle": "\u0e23\u0e16\u0e08\u0e31\u0e01\u0e23\u0e22\u0e32\u0e19\u0e22\u0e19\u0e15\u0e4c",
    "truck": "\u0e23\u0e16\u0e1a\u0e23\u0e23\u0e17\u0e38\u0e01",
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


def _set_tts_thread_priority():
    """Lower only the optional TTS worker on Windows.

    The detector and Qt event loop must remain responsive while ONNX is
    generating speech.  This is deliberately a thread priority change, not a
    process priority change, so video decoding and inference keep their normal
    scheduling priority.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentThread.restype = ctypes.c_void_p
        kernel32.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
        kernel32.SetThreadPriority.restype = ctypes.c_int
        kernel32.SetThreadPriority(
            kernel32.GetCurrentThread(), TTS_THREAD_PRIORITY
        )
    except (AttributeError, OSError, TypeError):
        # Audio must still work if the Windows API is unavailable (for example
        # under a compatibility layer).
        pass


def _load_vachana_voice(model_path):
    """Load VachanaTTS with a bounded ONNX CPU session.

    ``vachanatts.voice.Voice.load`` creates an ONNX Runtime session with the
    library defaults.  Those defaults can create many CPU workers and briefly
    starve OpenCV/Qt while the first close object is announced.  Construct the
    same dataclass with an explicitly bounded session instead of modifying the
    installed third-party package.
    """
    import onnxruntime
    from vachanatts.config import Config
    from vachanatts.voice import Voice

    with VACHANA_CONFIG.open("r", encoding="utf-8") as config_file:
        config = Config.from_dict(json.load(config_file))

    session_options = onnxruntime.SessionOptions()
    session_options.intra_op_num_threads = TTS_INTRA_OP_THREADS
    session_options.inter_op_num_threads = TTS_INTER_OP_THREADS
    try:
        # Do not keep CPU workers spinning between ONNX operators while the
        # detector and Qt event loop are trying to render the next frame.
        session_options.add_session_config_entry(
            "session.intra_op.allow_spinning", "0"
        )
        session_options.add_session_config_entry(
            "session.inter_op.allow_spinning", "0"
        )
    except (AttributeError, RuntimeError):
        pass
    execution_mode = getattr(onnxruntime, "ExecutionMode", None)
    sequential_mode = getattr(execution_mode, "ORT_SEQUENTIAL", None)
    if sequential_mode is not None:
        session_options.execution_mode = sequential_mode

    return Voice(
        config=config,
        session=onnxruntime.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        ),
    )


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
        self._thai_voice_probe_running = False
        self._thai_voice_probe_lock = threading.Lock()
        self._speech_lock = threading.Lock()
        self._audio_state_lock = threading.Lock()
        self._audio_playing_until = 0.0
        self._speech_thread = None
        self._preload_thread = None
        self._vachana_voice = None
        self._pending_message = None
        self._tts_compute_active = threading.Event()
        self._live_source = False
        self._voice_model = "th_m_1"
        self._voice_volume = 100
        self._vachana_model = TTS_DIR / "voices" / "th_m_1.onnx"
        self._vachana_available = None
        self._voice_generation = 0

        _prune_wav_cache(
            TEMP_TTS_DIR / "segments", SEGMENT_CACHE_MAX_FILES
        )
        _prune_wav_cache(TEMP_TTS_DIR / "volume", VOLUME_CACHE_MAX_FILES)

        if sys.platform == "win32" and VOICE_DIR.is_dir():
            try:
                import ctypes

                self._mci = ctypes.windll.winmm.mciSendStringW
            except (ImportError, OSError, AttributeError):
                self._mci = None

    @property
    def available(self):
        if not VOICE_SPEECH_ENABLED:
            return False
        return (
            self._mci is not None
            or self._segment_voice_available()
            or (not SEGMENT_VOICE_ONLY and self._bundled_tts_available())
        )

    def _segment_voice_available(self):
        segment_dir = SEGMENT_VOICE_DIR / self._voice_model
        try:
            return segment_dir.is_dir() and any(segment_dir.glob("*.wav"))
        except OSError:
            return False

    def _bundled_tts_available(self):
        if SEGMENT_VOICE_ONLY:
            return False
        if self._vachana_model.is_file() and VACHANA_CONFIG.is_file():
            if self._vachana_available is None:
                try:
                    self._vachana_available = find_spec("vachanatts.voice") is not None
                except (ImportError, AttributeError, ValueError):
                    self._vachana_available = False
            if self._vachana_available:
                return True
        return TTS_EXE.is_file() or (PIPER_EXE.is_file() and PIPER_MODEL.is_file())

    def set_voice_model(self, model_name):
        model_name = str(model_name or "th_m_1").strip()
        if model_name != "th_m_1":
            model_name = "th_m_1"
        model_path = TTS_DIR / "voices" / f"{model_name}.onnx"
        segment_path = SEGMENT_VOICE_DIR / model_name
        if not model_path.is_file() and not segment_path.is_dir():
            model_name = "th_m_1"
            model_path = TTS_DIR / "voices" / "th_m_1.onnx"
        if model_name != self._voice_model:
            self._voice_generation += 1
            self.stop()
            self._voice_model = model_name
            self._vachana_model = model_path
            self._vachana_voice = None
            self._vachana_available = None

    @property
    def voice_model(self):
        return self._voice_model

    @property
    def tts_compute_busy(self):
        """Whether TTS is using CPU for model loading or audio generation."""
        return self._tts_compute_active.is_set()

    def set_muted(self, muted):
        self.muted = bool(muted)
        if self.muted:
            self.stop()

    def set_enabled(self, enabled):
        """Toggle this channel from the alert settings screen."""
        self.enabled = bool(enabled) and VOICE_SPEECH_ENABLED
        if not self.enabled:
            self.stop()

    def set_volume(self, volume):
        """Set voice loudness for generated WAV and legacy MCI playback."""
        try:
            volume = int(round(float(volume)))
        except (TypeError, ValueError):
            volume = 100
        self._voice_volume = max(0, min(100, volume))
        if self._voice_volume <= 0:
            self.stop()

    @property
    def voice_volume(self):
        return self._voice_volume

    def set_live_source(self, active):
        """Avoid cache-miss TTS synthesis while a live source is playing."""
        self._live_source = bool(active)
        if self._live_source:
            self._pending_message = None

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
        if (
            not self.available
            or self.muted
            or not self.enabled
            or self._voice_volume <= 0
        ):
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

        ready_segments = segment_paths(
            SEGMENT_VOICE_DIR / self._voice_model, label, status, distance
        )
        if ready_segments:
            self._play_segmented_tts(ready_segments)
            return

        if self._live_source:
            # A cache hit is just WAV playback. A cache miss would load or
            # synthesize ONNX audio and can take several seconds on CPU, so
            # never start that work while video is live.
            cached_path = self._tts_cache_path(message)
            if self._valid_tts_audio(cached_path):
                self._play_cached_tts(cached_path)
                return
            self._play_legacy_danger_clip(label, status)
            return

        if SEGMENT_VOICE_ONLY:
            # Missing generated segments are an installation problem. Do not
            # fall back to ONNX here: loading/synthesizing speech can make the
            # detector compete for CPU and appear to freeze the video.
            self._play_legacy_danger_clip(label, status)
            return

        # Prefer a self-contained offline TTS engine shipped beside the app.
        # This is the path used on other computers, so it does not depend on
        # Windows having a Thai speech voice installed.
        if self._play_bundled_tts(message):
            return

        # SAPI can synthesize the changing distance at runtime. It avoids
        # needing a separate audio file for every possible meter value.
        if sys.platform == "win32" and self._has_thai_voice is None:
            # SAPI voice discovery starts PowerShell. Never run that process
            # synchronously from the Qt frame/update callback.
            self._start_thai_voice_probe()
            return
        if sys.platform == "win32" and self._has_thai_voice:
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

        # The bundled legacy clips are explicitly recorded for DANGER and do
        # not contain a distance. Never play one for WARNING, otherwise a
        # warning can be announced with the wrong severity. WARNING remains
        # silent when dynamic Thai TTS is unavailable.
        if status != "DANGER":
            return
        path = VOICE_DIR / f"{label}_danger.mp3"
        if not path.is_file():
            path = VOICE_DIR / "generic_danger.mp3"
        if path.is_file():
            self._play(path)

    def _audio_is_playing(self):
        with self._audio_state_lock:
            return time.monotonic() < self._audio_playing_until

    @staticmethod
    def _wav_duration(path):
        try:
            with wave.open(str(path), "rb") as audio:
                rate = audio.getframerate()
                return audio.getnframes() / rate if rate > 0 else 0.0
        except (OSError, ValueError, wave.Error):
            return 0.0

    def _play_segmented_tts(self, paths):
        """Concatenate pre-generated WAV words without loading an ONNX model."""
        if (
            sys.platform != "win32"
            or not paths
            or self._voice_volume <= 0
            or self._speech_lock.locked()
            or self._audio_is_playing()
        ):
            return

        cache_name = (
            f"{self._voice_model}_{SEGMENT_CACHE_VERSION}_"
            + "_".join(path.stem for path in paths)
            + ".wav"
        )
        cache_path = TEMP_TTS_DIR / "segments" / cache_name
        generation = self._voice_generation

        def play_segments():
            with self._speech_lock:
                try:
                    import wave
                    import winsound

                    if not self._valid_tts_audio(cache_path):
                        self._concat_wavs(paths, cache_path)
                        _prune_wav_cache(
                            TEMP_TTS_DIR / "segments",
                            SEGMENT_CACHE_MAX_FILES,
                        )
                    if self._valid_tts_audio(cache_path):
                        play_path = self._scaled_wav_path(cache_path)
                        if play_path is None:
                            return
                        duration = self._wav_duration(play_path)
                        with self._audio_state_lock:
                            if generation != self._voice_generation:
                                return
                            if time.monotonic() < self._audio_playing_until:
                                return
                            # Keep the application responsive with async
                            # playback, but reserve the full WAV duration so
                            # another alert cannot cut the final word short.
                            winsound.PlaySound(None, 0)
                            winsound.PlaySound(
                                str(play_path),
                                winsound.SND_FILENAME | winsound.SND_ASYNC,
                            )
                            self._audio_playing_until = (
                                time.monotonic() + duration + 0.05
                            )
                except (ImportError, OSError, RuntimeError, ValueError, wave.Error):
                    pass

        self._speech_thread = threading.Thread(
            target=play_segments, daemon=True
        )
        self._speech_thread.start()

    @staticmethod
    def _concat_wavs(paths, output_path):
        import wave

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(paths[0]), "rb") as first:
            params = first.getparams()
            sample_rate = first.getframerate()
            sample_width = first.getsampwidth()
            channels = first.getnchannels()
            first_audio = trim_pcm_frames(
                first.readframes(first.getnframes()),
                sample_width,
                channels,
            )
        if not first_audio:
            raise ValueError("first voice segment is silent")

        silence_frames = int(sample_rate * SEGMENT_GAP_MS / 1000)
        silence = b"\0" * silence_frames * sample_width * channels
        final_padding_frames = int(sample_rate * SEGMENT_FINAL_PADDING_MS / 1000)
        final_padding = b"\0" * final_padding_frames * sample_width * channels
        with wave.open(str(output_path), "wb") as output:
            output.setparams(params)
            output.writeframes(first_audio)
            for index, path in enumerate(paths[1:], start=1):
                with wave.open(str(path), "rb") as segment:
                    if (
                        segment.getframerate() != sample_rate
                        or segment.getsampwidth() != sample_width
                        or segment.getnchannels() != channels
                    ):
                        raise ValueError("voice segment formats do not match")
                    audio = trim_pcm_frames(
                        segment.readframes(segment.getnframes()),
                        sample_width,
                        channels,
                    )
                    if audio:
                        output.writeframes(silence)
                        output.writeframes(audio)
                        if index == len(paths) - 1 and final_padding:
                            # Leave a short clean tail after the final word.
                            # This prevents the last syllable (especially
                            # "เมตร") from sounding abruptly chopped.
                            output.writeframes(final_padding)

    def _play_legacy_danger_clip(self, label, status):
        if status != "DANGER" or self._mci is None:
            return
        path = VOICE_DIR / f"{label}_danger.mp3"
        if not path.is_file():
            path = VOICE_DIR / "generic_danger.mp3"
        if path.is_file():
            self._play(path)

    def _play_cached_tts(self, path):
        if (
            sys.platform != "win32"
            or not self._valid_tts_audio(path)
            or self._voice_volume <= 0
            or self._audio_is_playing()
        ):
            return
        if self._speech_lock.locked():
            return

        def play_cached():
            with self._speech_lock:
                try:
                    import winsound

                    play_path = self._scaled_wav_path(path)
                    if play_path is None:
                        return
                    duration = self._wav_duration(play_path)
                    with self._audio_state_lock:
                        if time.monotonic() < self._audio_playing_until:
                            return
                        winsound.PlaySound(None, 0)
                        winsound.PlaySound(
                            str(play_path),
                            winsound.SND_FILENAME | winsound.SND_ASYNC,
                        )
                        self._audio_playing_until = (
                            time.monotonic() + duration + 0.05
                        )
                except (ImportError, OSError, RuntimeError):
                    pass

        self._speech_thread = threading.Thread(target=play_cached, daemon=True)
        self._speech_thread.start()

    def _play_bundled_tts(self, message):
        if SEGMENT_VOICE_ONLY:
            return False
        if sys.platform != "win32":
            return False
        if not self._bundled_tts_available():
            return False

        # One utterance at a time. Generating/playing another WAV over the
        # previous one was the reason speech was being cut off.
        if self._speech_lock.locked():
            return True
        # Do not create a second ONNX session while the background preload is
        # still loading the model.  That race can briefly saturate every CPU
        # core exactly when the first close object is detected.
        if self._preload_thread and self._preload_thread.is_alive():
            self._pending_message = message
            return True
        if (
            self._vachana_voice is None
            and self._vachana_model.is_file()
            and VACHANA_CONFIG.is_file()
            and self._vachana_available is not False
        ):
            self._pending_message = message
            self.preload()
            return True
        self._speech_thread = threading.Thread(
            target=self._synthesize_and_play, args=(message,), daemon=True
        )
        self._speech_thread.start()
        return True

    def preload(self):
        """Load the bundled voice before the first alert is needed."""
        if (
            SEGMENT_VOICE_ONLY
            or sys.platform != "win32"
            or not self.enabled
            or self.muted
            or not self._vachana_model.is_file()
        ):
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
        loaded = False
        self._tts_compute_active.set()
        try:
            _set_tts_thread_priority()
            model_path = self._vachana_model
            loaded_voice = _load_vachana_voice(model_path)
            with self._speech_lock:
                if generation == self._voice_generation:
                    self._vachana_voice = loaded_voice
                    loaded = True
        except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
            self._vachana_available = False
        finally:
            self._tts_compute_active.clear()
            self._preload_thread = None
            # Model changes call ``preload`` explicitly after updating the
            # selected path. A camera close/mute must not restart a cancelled
            # preload while the video is being stopped.
            if generation == self._voice_generation and loaded and self._pending_message:
                pending_message = self._pending_message
                self._pending_message = None
                self._speech_thread = threading.Thread(
                    target=self._synthesize_and_play,
                    args=(pending_message,),
                    daemon=True,
                )
                self._speech_thread.start()

    def _synthesize_and_play(self, message):
        output_path = self._tts_cache_path(message)
        generation = self._voice_generation
        model_path = self._vachana_model
        _set_tts_thread_priority()
        self._tts_compute_active.set()
        with self._speech_lock:
            try:
                TEMP_TTS_DIR.mkdir(parents=True, exist_ok=True)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cached_audio = self._valid_tts_audio(output_path)
                if (
                    not cached_audio
                    and model_path.is_file()
                    and VACHANA_CONFIG.is_file()
                    and self._vachana_available is not False
                ):
                    try:
                        import wave
                        from vachanatts.config import SpeechConfig
                        if getattr(self, "_vachana_voice", None) is None:
                            self._vachana_voice = _load_vachana_voice(model_path)
                        with wave.open(str(output_path), "wb") as wav_file:
                            self._vachana_voice.synthesize_wav(
                                message,
                                wav_file,
                                SpeechConfig(
                                    volume=SPEECH_VOLUME,
                                    length_scale=SPEECH_LENGTH_SCALE,
                                ),
                            )
                    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
                        self._vachana_available = False
                        try:
                            output_path.unlink(missing_ok=True)
                        except OSError:
                            pass

                if not self._valid_tts_audio(output_path) and TTS_EXE.is_file():
                    subprocess.run(
                        [str(TTS_EXE), "--text", message, "--output", str(output_path)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=8,
                        check=True,
                    )
                elif (
                    not self._valid_tts_audio(output_path)
                    and PIPER_EXE.is_file()
                    and PIPER_MODEL.is_file()
                ):
                    subprocess.run(
                        [str(PIPER_EXE), "--model", str(PIPER_MODEL),
                         "--output_file", str(output_path)],
                        input=message, text=True,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        timeout=8, check=True,
                    )

                if self._valid_tts_audio(output_path):
                    self._trim_tts_cache(output_path)

                if self._valid_tts_audio(output_path) and generation == self._voice_generation:
                    # Playback itself is asynchronous from the application's
                    # point of view; allow detection to resume before the
                    # sound finishes.
                    self._tts_compute_active.clear()
                    import winsound

                    play_path = self._scaled_wav_path(output_path)
                    if play_path is None:
                        return
                    winsound.PlaySound(None, 0)
                    winsound.PlaySound(
                        str(play_path),
                        winsound.SND_FILENAME | winsound.SND_ASYNC,
                    )
            except (ImportError, OSError, RuntimeError, subprocess.SubprocessError):
                # Do not keep reporting a file-backed TTS engine as available
                # after its package/model failed. Future alerts can then use
                # SAPI or another bundled engine instead of failing silently.
                if model_path.is_file() and VACHANA_CONFIG.is_file():
                    self._vachana_available = False
            finally:
                self._tts_compute_active.clear()

    def _tts_cache_path(self, message):
        """Return a stable cache file so repeated alerts avoid ONNX work."""
        cache_key = hashlib.sha256(
            f"{self._voice_model}\0{message}".encode("utf-8")
        ).hexdigest()[:24]
        return TEMP_TTS_DIR / "cache" / f"{self._voice_model}_{cache_key}.wav"

    @staticmethod
    def _valid_tts_audio(path):
        try:
            return path.is_file() and path.stat().st_size > 44
        except OSError:
            return False

    @staticmethod
    def _trim_tts_cache(latest_path):
        try:
            cache_dir = latest_path.parent
            files = [
                path for path in cache_dir.glob("*.wav")
                if path.is_file()
            ]
            files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
            for old_path in files[TTS_CACHE_LIMIT:]:
                old_path.unlink(missing_ok=True)
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

    def _start_thai_voice_probe(self):
        with self._thai_voice_probe_lock:
            if self._has_thai_voice is not None or self._thai_voice_probe_running:
                return
            self._thai_voice_probe_running = True

        def probe():
            try:
                self._thai_voice_installed()
            finally:
                with self._thai_voice_probe_lock:
                    self._thai_voice_probe_running = False

        threading.Thread(target=probe, daemon=True).start()

    def _play(self, path):
        buf_size = 256
        import ctypes

        buf = ctypes.create_unicode_buffer(buf_size)
        # Close any clip still playing so the newest object always wins.
        self._mci(f"close {MCI_ALIAS}", buf, buf_size - 1, 0)
        self._mci(f'open "{path}" type mpegvideo alias {MCI_ALIAS}', buf, buf_size - 1, 0)
        volume = int(round(self._voice_volume * 10))
        self._mci(
            f"setaudio {MCI_ALIAS} volume to {volume}",
            buf,
            buf_size - 1,
            0,
        )
        self._mci(f"play {MCI_ALIAS}", buf, buf_size - 1, 0)

    def _scaled_wav_path(self, path):
        """Create/reuse a per-volume PCM copy without blocking the UI thread."""
        if self._voice_volume >= 100:
            return path
        if self._voice_volume <= 0:
            return None

        try:
            import struct

            source = Path(path)
            cache_key = hashlib.sha256(
                f"{source}\0{source.stat().st_mtime_ns}\0{self._voice_volume}".encode(
                    "utf-8"
                )
            ).hexdigest()[:24]
            output = TEMP_TTS_DIR / "volume" / f"{cache_key}_{self._voice_volume}.wav"
            if self._valid_tts_audio(output):
                return output
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".tmp.wav")
            with wave.open(str(source), "rb") as input_file:
                params = input_file.getparams()
                frames = input_file.readframes(input_file.getnframes())
                if input_file.getsampwidth() == 2:
                    count = len(frames) // 2
                    samples = list(struct.unpack(f"<{count}h", frames[: count * 2]))
                    scale = self._voice_volume / 100.0
                    samples = [
                        max(-32768, min(32767, int(round(sample * scale))))
                        for sample in samples
                    ]
                    frames = struct.pack(f"<{len(samples)}h", *samples)
                with wave.open(str(temporary), "wb") as output_file:
                    output_file.setparams(params)
                    output_file.writeframes(frames)
            os.replace(temporary, output)
            _prune_wav_cache(TEMP_TTS_DIR / "volume", VOLUME_CACHE_MAX_FILES)
            return output
        except (OSError, ValueError, wave.Error, struct.error):
            return path

    def stop(self):
        # Invalidate any bundled-TTS worker that is currently synthesizing so
        # it cannot start playback after the source has been closed or muted.
        self._voice_generation += 1
        self._pending_message = None
        self._reset_state()
        with self._audio_state_lock:
            self._audio_playing_until = 0.0
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
