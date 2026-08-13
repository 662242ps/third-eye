"""Non-blocking latest-frame capture for cameras and video files."""

import time
from threading import Event, Lock

import cv2
from PyQt5.QtCore import QThread, pyqtSignal

from vision.frame_utils import letterbox


class CaptureThread(QThread):
    """Read a source away from the Qt UI thread.

    Only the newest frame is retained.  This keeps a slow detector or a busy
    UI from building an unbounded queue and displaying stale camera frames.
    """

    error_ready = pyqtSignal(str)
    end_of_stream = pyqtSignal()

    def __init__(self, source, is_video=False, video_fps=30.0, output_size=None, parent=None):
        super().__init__(parent)
        self.source = source
        self.is_video = bool(is_video)
        self.video_fps = max(float(video_fps or 30.0), 1.0)
        self.output_size = tuple(output_size) if output_size else None
        self._stop_requested = Event()
        self._command_event = Event()
        self._lock = Lock()
        self._latest_frame = None
        self._latest_number = -1
        self._seek_request = None
        self._paused = False

    def run(self):
        capture = None
        try:
            capture = cv2.VideoCapture()
            opened = False
            if not self.is_video:
                # OpenCV supports read/open timeouts only for some backends,
                # and the parameterized overload is unavailable in older
                # builds. Try it before falling back to the regular open.
                params = []
                open_timeout = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
                read_timeout = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
                if open_timeout is not None:
                    params.extend([open_timeout, 3000])
                if read_timeout is not None:
                    params.extend([read_timeout, 1000])
                if params:
                    try:
                        opened = capture.open(self.source, cv2.CAP_ANY, params)
                    except (TypeError, cv2.error):
                        capture.release()
                        capture = cv2.VideoCapture()
                if not opened:
                    capture.release()
                    opened = capture.open(self.source)
            else:
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
            end_emitted = False

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
                    with self._lock:
                        self._latest_frame = None
                        self._latest_number = -1
                    continue

                if paused:
                    self._command_event.wait(0.05)
                    self._command_event.clear()
                    continue

                # For files, pace by the source timestamp.  If the UI or
                # detector falls behind, read and discard intermediate frames
                # so playback remains real-time instead of slowing down.
                if self.is_video:
                    target_frame = clock_frame + int(
                        (time.monotonic() - clock_started) * self.video_fps
                    )
                    if frame_number > target_frame:
                        time.sleep(0.002)
                        continue
                    # Skip decode work for frames that the UI can no longer
                    # display in time.  The detector receives the newest
                    # frame at the correct playback timestamp. ``read`` is
                    # used for the target itself; ``retrieve`` without a
                    # preceding grab would fail on the first frame.
                    while frame_number < target_frame:
                        if not capture.grab():
                            frame = None
                            ok = False
                            break
                        frame_number += 1
                    else:
                        ok, frame = capture.read()
                else:
                    ok, frame = capture.read()
                if not ok:
                    if self.is_video:
                        with self._lock:
                            self._paused = True
                        if not end_emitted:
                            end_emitted = True
                            self.end_of_stream.emit()
                        continue
                    self.error_ready.emit("ไม่สามารถอ่านเฟรมจากกล้องได้")
                    return

                if self.output_size:
                    frame = letterbox(frame, self.output_size)

                with self._lock:
                    self._latest_frame = frame
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
