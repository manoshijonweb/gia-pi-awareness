from __future__ import annotations
import logging
import queue
import threading
import time
import cv2
from .camera import Camera, ImageCamera, capture_once
from .commands import Command
from .common import FeatureError, Cancelled
from .distance import Calibration, describe_distance
from .speech import Speaker
from .tof import DistanceSensor

HELP = {
    "en": "Say scan, color, read, or distance. Say repeat to hear the last result. Say calibrate to set your step length. During long reading, say stop in a listening pause.",
    "hi": "कहें, देखो, रंग, पढ़ो, या दूरी। दोबारा सुनने के लिए कहें, दोहराओ। कदम की लंबाई बदलने के लिए कहें, कदम नापो। लंबे पाठ के बीच विराम में कहें, रुको।",
}

class Application:
    def __init__(self, cfg, image=None, no_speech: bool = False):
        self.cfg = cfg
        cv2.setNumThreads(1)
        self.speaker = Speaker(cfg.voice, silent=no_speech)
        self.camera = ImageCamera(image) if image else None
        self.sensor = DistanceSensor(cfg.distance)
        self.calibration = Calibration(cfg.distance.calibration_file, cfg.distance.default_step_mm)
        self.commands = queue.Queue(maxsize=2)
        self.cancel = threading.Event()
        self.done = threading.Event()
        self.listener = None
        self.detector = self.reader = None
        self.camera_error = ""
        self.feature_errors = {}
        self.last_response = None
        self.language_mode = cfg.response_language
        self.calibrating = False
        self.pending_step = None
        self.log = logging.getLogger(__name__)

    def start(self, voice: bool = True):
        # Power-aware startup: do not stream the camera or poll ToF in the idle state.
        # Optional one-time model warmups happen before the ready announcement so later
        # voice commands avoid first-use graph initialization latency.
        if self.camera is not None:
            try:
                self.camera.start()
            except Exception as exc:
                self.camera_error = str(exc)
                self.log.warning("Test-image camera unavailable: %s", exc)
        self.sensor.start()  # no-op for the on-demand VL53L0X profile
        warm = []
        power = getattr(self.cfg, "power", None)
        if getattr(power, "preload_detector", False):
            warm.append(("scan", self._detector))
        if getattr(power, "preload_ocr", False):
            warm.append(("read", self._reader))
        if self.cfg.preload_models:  # legacy compatibility: explicitly requested means both
            warm = [("scan", self._detector), ("read", self._reader)]
        for name, loader in warm:
            try:
                loader().warmup()
            except Exception as exc:
                self.feature_errors[name] = str(exc)
                self.log.warning("%s warmup failed: %s", name, exc)

        if voice:
            from .voice import CommandListener
            self.listener = CommandListener(
                self.cfg.voice, self.speaker, self.submit, self.cfg.log_recognized_text
            )
            # Do not let the microphone hear the boot announcement.  Say exactly the
            # requested phrase first, then enable capture.
            self.speaker.say(self.cfg.voice.ready_text_en, "en")
            self.speaker.wait_idle(timeout=20)
            self.listener.start()
        return self

    def _detector(self):
        if self.detector is None:
            from .objects import ObjectDetector
            self.detector = ObjectDetector(self.cfg.vision)
        return self.detector

    def _reader(self):
        if self.reader is None:
            from .ocr import TextReader
            self.reader = TextReader(self.cfg.vision)
        return self.reader

    def _frame(self):
        if self.camera_error:
            raise FeatureError(self.camera_error)
        if self.camera is not None:
            return self.camera.snapshot()
        # Normal Pi service path: power the CSI camera only for this command.
        return capture_once(self.cfg.camera)

    def submit(self, command: Command, ended: float | None = None):
        ended = time.monotonic() if ended is None else ended
        if command.intent in ("stop", "cancel"):
            self.cancel.set()
            self.speaker.stop()
            while True:
                try:
                    self.commands.get_nowait()
                except queue.Empty:
                    break
            if command.intent == "stop":
                return
        try:
            self.commands.put_nowait((command, ended))
        except queue.Full:
            self.log.warning("Busy; extra command dropped instead of accumulating stale work")

    def process(self, command: Command, ended: float | None = None) -> str:
        ended = time.monotonic() if ended is None else ended
        start = time.monotonic()
        self.cancel.clear()
        self.speaker.stop()  # A newly accepted command supersedes a previous spoken result.
        language = command.language if self.language_mode == "auto" else self.language_mode
        intent = command.intent
        try:
            if intent == "stop":
                self.cancel.set()
                return ""
            if intent == "cancel":
                self.calibrating, self.pending_step = False, None
                response = "रद्द कर दिया।" if language == "hi" else "Cancelled."
            elif intent == "help":
                response = HELP[language]
            elif intent == "repeat":
                if self.last_response:
                    response, language = self.last_response
                else:
                    response = "दोहराने के लिए कुछ नहीं है।" if language == "hi" else "There is no previous result."
            elif intent.startswith("language_"):
                self.language_mode = intent.removeprefix("language_")
                if self.language_mode == "hi":
                    response, language = "अब हिंदी में बोलूंगा।", "hi"
                elif self.language_mode == "en":
                    response, language = "I will speak English.", "en"
                else:
                    response = "भाषा अपने आप चुनूंगा।" if language == "hi" else "I will match your command language."
            elif intent == "calibrate":
                self.calibrating, self.pending_step = True, None
                response = ("सहायक से अपनी सामान्य कदम लंबाई नपवाएं। सेंटीमीटर में कहें, कदम की लंबाई पचहत्तर। या कहें, रद्द करो।" if language == "hi"
                            else "Have a helper measure your normal step length. Say step length seventy five, in centimeters, or say cancel.")
            elif intent == "set_step":
                if not self.calibrating:
                    response = "पहले कदम नापो कहें।" if language == "hi" else "Say calibrate first."
                else:
                    self.pending_step = Calibration.validate(command.value)
                    cm = self.pending_step / 10
                    response = (f"कदम की लंबाई {cm:g} सेंटीमीटर रखूं? पुष्टि या रद्द करो कहें।" if language == "hi"
                                else f"Set step length to {cm:g} centimeters? Say confirm or cancel.")
            elif intent == "confirm":
                if self.calibrating and self.pending_step is not None:
                    self.calibration.save(self.pending_step, "voice_confirmed")
                    self.calibrating, self.pending_step = False, None
                    response = "कदम की लंबाई सहेज दी।" if language == "hi" else "Step length saved."
                else:
                    response = "पुष्टि करने के लिए कोई बदलाव नहीं है।" if language == "hi" else "There is no calibration change to confirm."
            elif intent == "distance":
                response = describe_distance(self.sensor.get(), self.calibration.step_mm, language,
                    max_age_s=self.cfg.distance.max_age_s, minimum_mm=self.cfg.distance.min_distance_mm,
                    maximum_mm=self.cfg.distance.max_distance_mm)
            elif intent == "color":
                from .colors import identify_color, describe_color
                result = identify_color(self._frame(), self.cfg.vision)
                response = describe_color(result, language, self.cfg.vision.color_min_dominance)
            elif intent == "scan":
                from .objects import describe_objects
                if intent in self.feature_errors:
                    raise FeatureError(self.feature_errors[intent])
                response = describe_objects(self._detector().detect(self._frame()), language, self.cfg.vision.max_object_names)
            elif intent == "read":
                from .ocr import describe_text
                if intent in self.feature_errors:
                    raise FeatureError(self.feature_errors[intent])
                result = self._reader().read(self._frame(), self.cancel)
                response = describe_text(result, language)
            else:
                raise FeatureError(f"Unsupported command intent: {intent}")
            if self.cancel.is_set():
                raise Cancelled("Discarding cancelled result")
        except Cancelled:
            return ""
        except Exception as exc:
            self.log.exception("Feature %s unavailable: %s", intent, exc)
            response = ("यह सुविधा अभी उपलब्ध नहीं है। उपकरण की सेटिंग जांचें।" if language == "hi"
                        else "This feature is currently unavailable. Check the device setup.")
        self.log.info("METRIC intent=%s feature_compute_s=%.3f end_of_speech_to_result_ready_s=%.3f",
                      intent, time.monotonic()-start, time.monotonic()-ended)
        if intent not in ("repeat", "confirm", "cancel", "calibrate", "set_step"):
            self.last_response = response, language
        self.speaker.say(response, language, ended)
        return response

    def run(self):
        while not self.done.is_set():
            try:
                command, ended = self.commands.get(timeout=0.2)
            except queue.Empty:
                if self.listener and not self.listener.thread.is_alive():
                    raise FeatureError("Voice listener stopped; inspect logs and microphone settings")
                continue
            self.process(command, ended)

    def close(self):
        self.done.set()
        self.cancel.set()
        if self.listener:
            self.listener.close()
        self.speaker.close()
        self.sensor.close()
        if self.camera is not None:
            self.camera.close()
