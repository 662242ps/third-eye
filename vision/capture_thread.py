"""Non-blocking latest-frame capture for cameras and video files."""

import math
import time
from threading import Event, Lock

import cv2
from PyQt5.QtCore import QThread, pyqtSignal

from vision.frame_utils import letterbox, letterbox_with_meta


class CaptureThread(QThread):
    """Read a source away from the Qt UI thread.

    Only the newest frame is retained.  This keeps a slow detector or a busy
    UI from building an unbounded queue and displaying stale camera frames.
    """

    error_ready = pyqtSignal(str)
    end_of_stream = pyqtSignal()
    VIDEO_READ_RETRIES = 5
    MAX_GRAB_PER_CYCLE = 4

    def __init__(
        self,
        source,
        is_video=False,
        video_fps=30.0,
        output_size=None,
        model_size=(640, 640),
        parent=None,
    ):
        super().__init__(parent)
        self.source = source
        self.is_video = bool(is_video)
        try:
            video_fps = float(video_fps)
        except (TypeError, ValueError):
            video_fps = 30.0
        if not math.isfinite(video_fps) or video_fps <= 0:
            video_fps = 30.0
        self.video_fps = max(video_fps, 1.0)
        self.output_size = tuple(output_size) if output_size else None
        self.model_size = tuple(model_size) if model_size else None
        self._stop_requested = Event()
        self._command_event = Event()
        self._lock = Lock()
        self._latest_frame = None
        self._latest_model_frame = None
        self._latest_model_transform = None
        self._latest_number = -1
        self._seek_request = None
        self._paused = False

    def run(self):
        capture = None
        try:
            capture = cv2.VideoCapture()
            # Timeout parameters are backend-dependent. Try them for both
            # cameras and files, then fall back to the normal overload when a
            # particular OpenCV backend does not support them.
            params = []
            open_timeout = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
            read_timeout = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
            if open_timeout is not None:
                params.extend([open_timeout, 3000])
            if read_timeout is not None:
                params.extend([read_timeout, 1000])
            opened = False
            if params:
                try:
                    opened = capture.open(self.source, cv2.CAP_ANY, params)
                except (TypeError, cv2.error):
                    capture.release()
                    capture = cv2.VideoCapture()
            if not opened:
                capture.release()
                opened = capture.open(self.source)

            if not opened or not capture.isOpened():
                self.error_ready.emit(f"ไม่สามารถเปิดแหล่งภาพได้: {self.source}")
                return

            # Some camera backends ignore this setting, but the ones that
            # support it avoid displaying frames that are already stale.
            if not self.is_video:
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # Backends such as FFmpeg/DSHOW may support these timeout
                # properties. Unsupported backends simply ignore them.
                open_timeout = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
                read_timeout = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
                if open_timeout is not None:
                    capture.set(open_timeout, 3000)
                if read_timeout is not None:
                    capture.set(read_timeout, 1000)

            frame_number = 0
            clock_frame = 0
            clock_started = time.monotonic()
            video_total_frames = 0
            if self.is_video:
                try:
                    video_total_frames = max(
                        0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    )
                except (TypeError, ValueError, OverflowError):
                    video_total_frames = 0
            end_emitted = False
            read_failures = 0

            while not self._stop_requested.is_set():
                with self._lock:
                    seek_request = self._seek_request
                    self._seek_request = None
                    paused = self._paused

                if seek_request is not None:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(seek_request)))
                    frame_number = max(0, int(seek_request))
                    clock_frame = frame_number
                    clock_started = time.monotonic()
                    read_failures = 0
                    with self._lock:
                        self._latest_frame = None
                        self._latest_model_frame = None
                        self._latest_model_transform = None
                        self._latest_number = -1
                    continue

                if paused:
                    self._command_event.wait(0.05)
                    self._command_event.clear()
                    continue

                # Pace playback against the source timestamp. When inference
                # falls behind, skip only a bounded number of frames per loop.
                # This reduces decode/CPU pressure without creating the long
                # grab burst that previously made the preview look frozen.
                if self.is_video:
                    now = time.monotonic()
                    target_frame = clock_frame + int(
                        (now - clock_started) * self.video_fps
                    )
                    if frame_number > target_frame:
                        # Do not busy-wait every 2 ms while the next source
                        # frame is not due yet. That loop consumed CPU even
                        # when no frame could be displayed and competed with
                        # Qt/YOLO on smaller machines.
                        next_frame_at = clock_started + (
                            (frame_number - clock_frame) / self.video_fps
                        )
                        wait_seconds = max(0.001, next_frame_at - now)
                        self._command_event.wait(min(wait_seconds, 0.05))
                        self._command_event.clear()
                        continue
                    skipped = 0
                    while (
                        frame_number < target_frame
                        and skipped < self.MAX_GRAB_PER_CYCLE
                    ):
                        if not capture.grab():
                            ok, frame = False, None
                            break
                        frame_number += 1
                        skipped += 1
                    else:
                        ok, frame = capture.read()
                else:
                    ok, frame = capture.read()
                if not ok:
                    if self.is_video:
                        try:
                            position = capture.get(cv2.CAP_PROP_POS_FRAMES)
                        except (TypeError, ValueError, OverflowError):
                            position = -1
                        at_end = (
                            video_total_frames > 0
                            and position >= video_total_frames - 1
                        )
                        if at_end:
                            with self._lock:
                                self._paused = True
                            if not end_emitted:
                                end_emitted = True
                                self.end_of_stream.emit()
                            continue
                        read_failures += 1
                        if read_failures <= self.VIDEO_READ_RETRIES:
                            self._command_event.wait(0.05)
                            self._command_event.clear()
                            continue
                        if video_total_frames <= 0:
                            with self._lock:
                                self._paused = True
                            if not end_emitted:
                                end_emitted = True
                                self.end_of_stream.emit()
                            continue
                        # Try to recover from a transient decoder failure at
                        # the current frame before giving up on the source.
                        capture.set(cv2.CAP_PROP_POS_FRAMES, max(frame_number, 0))
                        read_failures = 0
                        continue
                    self.error_ready.emit("ไม่สามารถอ่านเฟรมจากกล้องได้")
                    return

                read_failures = 0

                original_frame = frame
                display_frame = (
                    letterbox(original_frame, self.output_size)
                    if self.output_size
                    else original_frame
                )
                model_frame = None
                model_transform = None
                if self.model_size:
                    model_frame, model_transform = letterbox_with_meta(
                        original_frame, self.model_size
                    )
                    display_transform = None
                    if self.output_size:
                        _, display_transform = letterbox_with_meta(
                            original_frame, self.output_size
                        )
                    else:
                        display_transform = {
                            "scale": 1.0,
                            "offset_x": 0,
                            "offset_y": 0,
                            "source_w": original_frame.shape[1],
                            "source_h": original_frame.shape[0],
                        }
                    model_transform = {
                        "model": model_transform,
                        "display": display_transform,
                    }

                with self._lock:
                    self._latest_frame = display_frame
                    self._latest_model_frame = model_frame
                    self._latest_model_transform = model_transform
                    self._latest_number = frame_number
                frame_number += 1
                end_emitted = False
        except Exception as error:
            self.error_ready.emit(f"Capture error: {error}")
        finally:
            if capture is not None:
                capture.release()

    def get_latest(self):
        """Return the newest published frame and its sequence number.

        The capture worker replaces the frame reference instead of mutating a
        published ndarray, so an extra full-resolution copy is unnecessary.
        """
        with self._lock:
            if self._latest_frame is None:
                return None, -1
            return self._latest_frame, self._latest_number

    def get_latest_model(self):
        """Return the newest exact model-size frame and its display mapping."""
        with self._lock:
            if self._latest_model_frame is None:
                return None, -1, None
            return (
                self._latest_model_frame,
                self._latest_number,
                self._latest_model_transform,
            )

    def pause(self):
        with self._lock:
            self._paused = True
        self._command_event.set()

    def resume(self):
        with self._lock:
            self._paused = False
        self._command_event.set()

    def seek(self, frame_number):
        with self._lock:
            self._seek_request = max(0, int(frame_number))
        self._command_event.set()

    def stop(self, timeout=1000):
        self._stop_requested.set()
        self._command_event.set()
        if self.isRunning():
            return self.wait(timeout)
        return True
