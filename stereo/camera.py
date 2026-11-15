"""Opening the ELP cameras and reading frames from one or two at a time."""
import sys
import threading
import time

import cv2


def _backend():
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_V4L2


def open_camera(index, width, height, fps, warmup=10):
    """Open a camera in MJPG at exactly this resolution, or raise if it won't cooperate."""
    cap = cv2.VideoCapture(index, _backend())
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {index}. "
                           "Run 00_list_cameras.py to see which indices exist.")

    # Set MJPG before the size and fps, otherwise a lot of UVC cameras quietly
    # fall back to YUYV, which is far too slow over USB 2.0.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for _ in range(warmup):  # give auto-exposure a moment to settle
        cap.read()

    got = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if got != (width, height):
        cap.release()
        raise RuntimeError(f"Camera {index} gave us {got[0]}x{got[1]} "
                           f"instead of the {width}x{height} we asked for.")
    return cap


class ThreadedCamera:
    """Keeps reading frames in the background so read() always returns the latest one.

    That way a slow camera never holds up the other one. The timestamps are when
    the frame reached the PC, not when it was exposed, but that's still good
    enough to measure how far apart the two cameras are.
    """

    def __init__(self, index, width, height, fps):
        self.index = index
        self.cap = open_camera(index, width, height, fps)
        self._lock = threading.Lock()
        self._frame, self._t, self._count = None, 0.0, 0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ok, frame = self.cap.read()
            t = time.perf_counter()
            if ok:
                with self._lock:
                    self._frame, self._t = frame, t
                    self._count += 1
            else:
                time.sleep(0.005)

    def read(self):
        """Returns (latest frame or None, its timestamp, frames received so far)."""
        with self._lock:
            f = None if self._frame is None else self._frame.copy()
            return f, self._t, self._count

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
