from __future__ import annotations
import importlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


def memory_mb() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


def doctor(cfg, deep: bool = False, hardware: bool = False) -> int:
    failures, warnings = [], []
    def ok(text):
        print("PASS  " + text)
    def fail(text):
        failures.append(text); print("FAIL  " + text)
    def warn(text):
        warnings.append(text); print("WARN  " + text)

    print(f"PI_AWARENESS | Python {platform.python_version()} | {platform.machine()} | {platform.system()}")
    if sys.maxsize <= 2**32:
        fail("A 64-bit operating system/Python is required")
    else:
        ok("64-bit Python")
    if not (3, 11) <= sys.version_info[:2] <= (3, 13):
        warn("Installer targets Python 3.11-3.13")
    if platform.machine() not in ("aarch64", "arm64"):
        warn("This is not an ARM64 Pi; desktop tests are not Pi benchmarks")
    memory = memory_mb()
    if memory is not None:
        print(f"INFO  RAM: {memory} MiB")
        if memory < 3500 and cfg.vision.detector == "yolo26m":
            warn("YOLO26m + bilingual ASR/OCR is best tested on 4 GB+ RAM")
    print("INFO  Idle policy: microphone/VAD/ASR only; CSI camera and VL53L0X are on-demand")

    dependencies = ["numpy", "cv2", "onnxruntime", "vosk"]
    if cfg.voice.backend == "sounddevice":
        dependencies.append("sounddevice")
    if cfg.distance.enabled and getattr(cfg.distance, "driver", "vl53l0x") == "vl53l0x":
        dependencies.extend(["adafruit_vl53l0x", "board", "busio"])
    for name in dependencies:
        try:
            module = importlib.import_module(name)
            ok(f"{name} {getattr(module, '__version__', '')}")
        except Exception as exc:
            fail(f"{name}: {exc}")

    if cfg.camera.backend != "usb":
        try:
            importlib.import_module("picamera2")
            ok("Picamera2 available for Raspberry Pi CSI camera")
        except Exception as exc:
            fail(f"Picamera2: {exc}")

    for name in ("espeak-ng", "aplay", "arecord"):
        (ok if shutil.which(name) else fail)(f"Executable {name}")

    i2s_device = None
    if cfg.voice.backend == "inmp441_i2s":
        from hardware.i2s_common import find_i2s_device, overlay_is_configured
        configured = overlay_is_configured()
        (ok if configured else fail)("dtoverlay=googlevoicehat-soundcard configured")
        try:
            i2s_device = find_i2s_device()
            ok(f"Shared I2S ALSA card: {i2s_device}")
        except Exception as exc:
            if configured and not hardware:
                warn(f"I2S ALSA card not active yet (reboot may be required): {exc}")
            else:
                fail(f"I2S audio card: {exc}")

    if shutil.which("espeak-ng"):
        try:
            from .speech import resolve_voices, resolve_output_device
            for lang, selected in resolve_voices(cfg.voice).items():
                ok(f"TTS {lang}: {selected}")
            if cfg.voice.speaker_device == "i2s-auto" and i2s_device is None and not hardware:
                print("INFO  Speaker PCM: i2s-auto (will resolve after reboot)")
            else:
                print(f"INFO  Speaker PCM: {resolve_output_device(cfg.voice.speaker_device) or 'OS default'}")
        except Exception as exc:
            fail(f"TTS/audio inventory: {exc}")

    required = [cfg.vision.object_model, cfg.vision.ocr_det, cfg.vision.ocr_rec]
    if cfg.vision.ocr_use_cls:
        required.append(cfg.vision.ocr_cls)
    for path in required:
        (ok if path.is_file() and path.stat().st_size > 100000 else fail)(f"Model {path.name}")
    for lang in cfg.voice.languages:
        path = cfg.voice.asr_hi if lang == "hi" else cfg.voice.asr_en
        (ok if (path / "am/final.mdl").is_file() else fail)(f"Speech model {path.name}")

    from .distance import Calibration
    calibration = Calibration(cfg.distance.calibration_file, cfg.distance.default_step_mm)
    print(f"INFO  Step length: {calibration.step_mm:g} mm; "
          f"{'user calibration' if calibration.calibrated else 'DEFAULT'}")

    if deep:
        try:
            from .objects import ObjectDetector
            detector = ObjectDetector(cfg.vision); detector.warmup()
            ok(f"Detector smoke test: {cfg.vision.detector}, {detector.output_format}, {detector.h}x{detector.w}")
            print(f"INFO  Warm-up timing: {detector.last_timings}")
            del detector
        except Exception as exc:
            fail(f"Detector smoke test: {exc}")
        try:
            from .ocr import TextReader
            reader = TextReader(cfg.vision); reader.warmup()
            ok(f"OCR graphs/dictionary smoke test: {len(reader.characters)} entries")
            del reader
        except Exception as exc:
            fail(f"OCR smoke test: {exc}")
        try:
            import vosk
            from .voice import vocabulary_grammar
            from .commands import parse_command
            vosk.SetLogLevel(-1)
            for lang in cfg.voice.languages:
                path = cfg.voice.asr_hi if lang == "hi" else cfg.voice.asr_en
                model = vosk.Model(str(path))
                phrases = vocabulary_grammar(model, lang, cfg.voice.require_wake_prefix)
                covered = {c.intent for phrase in phrases
                           if (c := parse_command(phrase, lang, cfg.voice.require_wake_prefix))}
                missing = {"scan", "color", "read", "distance"} - covered
                (warn if missing else ok)(f"{lang} voice vocabulary; missing core intents: {sorted(missing)}")
                rec = (vosk.KaldiRecognizer(model, 16000, json.dumps(phrases, ensure_ascii=False))
                       if cfg.voice.use_grammar else vosk.KaldiRecognizer(model, 16000))
                rec.AcceptWaveform(bytes(3200)); rec.FinalResult()
                del rec, model
        except Exception as exc:
            fail(f"ASR model/grammar smoke test: {exc}")

    if hardware:
        if cfg.voice.backend == "inmp441_i2s":
            try:
                from hardware.inmp441 import Microphone
                import numpy as np
                pcm = Microphone().speech_pcm(1.0, gain=cfg.voice.i2s_gain)
                samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
                rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
                ok(f"INMP441 captured 1 s through I2S; post-conversion RMS={rms:.1f}")
                if rms == 0:
                    warn("Microphone capture is digital silence; check L/R=GND and I2S wiring")
            except Exception as exc:
                fail(f"INMP441: {exc}")
        else:
            try:
                from .voice import choose_input
                import sounddevice as sd
                import numpy as np
                rate, info = choose_input(cfg.voice)
                with sd.RawInputStream(device=cfg.voice.microphone, samplerate=rate,
                                       channels=cfg.voice.channels, dtype="int16") as stream:
                    data, _ = stream.read(int(rate * 0.2))
                a = np.frombuffer(data, np.int16).astype(np.float32)
                ok(f"Microphone opened: {info['name']}; RMS={float(np.sqrt(np.mean(a*a))):.1f}")
            except Exception as exc:
                fail(f"Microphone: {exc}")

        camera = None
        try:
            from .camera import capture_once
            frame = capture_once(cfg.camera)
            ok(f"Pi camera frame {frame.shape}; image was not saved")
        except Exception as exc:
            fail(f"Camera: {exc}")

        if cfg.distance.enabled:
            try:
                from .tof import VL53L0X
                sensor = VL53L0X(cfg.distance)
                try:
                    result = sensor.sample()
                finally:
                    sensor.close()
                if result.valid:
                    ok(f"VL53L0X valid reading: {result.mm:g} mm")
                else:
                    warn(f"VL53L0X responded but no valid target: {result.reason}")
            except Exception as exc:
                fail(f"VL53L0X: {exc}")

        if shutil.which("vcgencmd"):
            for argument in ("measure_temp", "get_throttled"):
                try:
                    print("INFO  " + subprocess.check_output(["vcgencmd", argument], text=True).strip())
                except Exception:
                    pass

    print(f"\n{len(failures)} failure(s), {len(warnings)} warning(s).")
    return 1 if failures else 0
