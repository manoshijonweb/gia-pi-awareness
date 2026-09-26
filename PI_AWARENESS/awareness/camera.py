from __future__ import annotations
import logging
import threading
import time
import cv2
import numpy as np
from .common import FeatureError


def prepare_frame(frame: np.ndarray, cfg) -> np.ndarray:
    if frame is None or frame.size == 0:
        raise FeatureError("Empty camera frame")
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim != 3:
        raise FeatureError(f"Unexpected camera frame shape: {frame.shape}")
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.shape[2] != 3:
        raise FeatureError(f"Unexpected camera channel count: {frame.shape}")
    if cfg.rotation:
        rotations = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                     270: cv2.ROTATE_90_COUNTERCLOCKWISE}
        if cfg.rotation not in rotations:
            raise FeatureError("camera.rotation must be 0, 90, 180, or 270")
        frame = cv2.rotate(frame, rotations[cfg.rotation])
    if cfg.mirror:
        frame = cv2.flip(frame, 1)
    return np.ascontiguousarray(frame)

class PicameraBackend:
    def __init__(self, cfg):
        from picamera2 import Picamera2
        self.camera = Picamera2(int(cfg.device) if isinstance(cfg.device, int) else 0)
        try:
            # Picamera2's RGB888 is byte-ordered BGR in the returned numpy array.
            # That is the order OpenCV expects. Do NOT add an extra R/B swap here.
            config = self.camera.create_video_configuration(
                main={"size": (cfg.width, cfg.height), "format": "RGB888"},
                controls={"FrameRate": cfg.fps}, buffer_count=3, queue=False)
            self.camera.configure(config)
            if cfg.autofocus and "AfMode" in self.camera.camera_controls:
                from libcamera import controls
                self.camera.set_controls({"AfMode": controls.AfModeEnum.Continuous})
            self.camera.start()
        except Exception:
            self.camera.close()
            raise

    def read(self):
        return self.camera.capture_array("main")

    def close(self):
        self.camera.stop()
        self.camera.close()

class USBBackend:
    def __init__(self, cfg):
        self.cap = cv2.VideoCapture(cfg.device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap.release()
            raise FeatureError(f"Cannot open USB camera {cfg.device}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self.cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cfg.autofocus:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # unsupported controls may safely return False

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            raise FeatureError("USB camera did not supply a frame")
        return frame

    def close(self):
        self.cap.release()

class Camera:
    """Continuously drain the camera. Commands use a recent frame, never a stale FIFO."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.condition = threading.Condition()
        self.done = threading.Event()
        self.latest = None
        self.at = 0.0
        self.error = "Camera has not started"
        self.backend = None
        self.thread = None

    def start(self):
        choices = {"auto": [PicameraBackend, USBBackend], "picamera2": [PicameraBackend], "usb": [USBBackend]}
        if self.cfg.backend not in choices:
            raise FeatureError("camera.backend must be auto, picamera2, or usb")
        failures = []
        for factory in choices[self.cfg.backend]:
            try:
                self.backend = factory(self.cfg)
                break
            except Exception as exc:
                failures.append(f"{factory.__name__}: {exc}")
        if not self.backend:
            raise FeatureError("; ".join(failures))
        logging.getLogger(__name__).info("Camera backend: %s", type(self.backend).__name__)
        self.thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self.thread.start()
        try:
            self.snapshot(timeout=self.cfg.startup_timeout_s)
        except Exception:
            self.close()
            raise
        return self

    def _run(self):
        while not self.done.is_set():
            try:
                frame = self.backend.read()
                frame = prepare_frame(frame, self.cfg)
                with self.condition:
                    self.latest = frame.copy()
                    self.at = time.monotonic()
                    self.error = ""
                    self.condition.notify_all()
            except Exception as exc:
                with self.condition:
                    self.latest = None
                    self.error = str(exc)
                    self.condition.notify_all()
                self.done.wait(0.2)

    def snapshot(self, timeout: float = 1.5) -> np.ndarray:
        deadline = time.monotonic() + timeout
        with self.condition:
            while time.monotonic() < deadline:
                if self.latest is not None and time.monotonic() - self.at <= self.cfg.max_frame_age_s:
                    return self.latest.copy()
                self.condition.wait(min(0.1, max(0, deadline-time.monotonic())))
        raise FeatureError(self.error or "No fresh camera frame")

    def close(self):
        self.done.set()
        if self.thread:
            self.thread.join(timeout=1.5)
        if self.backend:
            try:
                self.backend.close()
            except Exception:
                logging.getLogger(__name__).exception("Camera shutdown failed")
        if self.thread:
            self.thread.join(timeout=1)

def capture_once(cfg) -> np.ndarray:
    """Open the camera only for one command, capture a fresh frame, then power it down.

    This is the default Pi service path.  It costs some command latency but avoids
    a continuously streaming CSI sensor/ISP while the device is only listening.
    """
    choices = {"auto": [PicameraBackend, USBBackend], "picamera2": [PicameraBackend],
               "usb": [USBBackend]}
    if cfg.backend not in choices:
        raise FeatureError("camera.backend must be auto, picamera2, or usb")
    failures = []
    for factory in choices[cfg.backend]:
        backend = None
        try:
            backend = factory(cfg)
            if getattr(cfg, "warmup_s", 0) > 0:
                time.sleep(cfg.warmup_s)
            # Discard an initial frame after sensor start and use the next fresh frame.
            first = backend.read()
            frame = backend.read() if getattr(cfg, "warmup_s", 0) > 0 else first
            return prepare_frame(frame, cfg)
        except Exception as exc:
            failures.append(f"{factory.__name__}: {exc}")
        finally:
            if backend is not None:
                try:
                    backend.close()
                except Exception:
                    logging.getLogger(__name__).exception("On-demand camera shutdown failed")
    raise FeatureError("; ".join(failures))


class ImageCamera:
    """Explicit test input only; never selected automatically after a camera failure."""
    def __init__(self, path):
        self.image = cv2.imread(str(path))
        if self.image is None:
            raise FeatureError(f"Cannot read image: {path}")
    def snapshot(self, **kwargs):
        return self.image.copy()
    def start(self):
        return self
    def close(self):
        pass
