#!/usr/bin/env python3
"""Local device entrypoint. Despite its name, this does NOT open an HTTP server."""
from __future__ import annotations
import os
# Set these before loading numerical/model libraries.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import argparse
from pathlib import Path
import signal
import sys
import time
from settings import CONFIG
from awareness.detector_profiles import PROFILES, apply_profile


def main() -> int:
    parser = argparse.ArgumentParser(description="Fully offline Pi 5 environmental-awareness prototype")
    parser.add_argument("--detector", choices=tuple(PROFILES), help="Override detector profile for this run; weights must already be provisioned")
    parser.add_argument("--doctor", action="store_true", help="Check local installation without speaking")
    parser.add_argument("--deep", action="store_true", help="Doctor: execute model warmups and validate Vosk grammar")
    parser.add_argument("--hardware", action="store_true", help="Doctor: open microphone, camera and ToF sensor")
    parser.add_argument("--list-audio", action="store_true")
    parser.add_argument("--text", action="store_true", help="Type commands instead of using a microphone")
    parser.add_argument("--once", help="Run one written command, e.g. 'scan' or 'read'")
    parser.add_argument("--image", type=Path, help="Explicit local image input instead of a live camera")
    parser.add_argument("--no-speech", action="store_true", help="Print responses instead of playing audio")
    parser.add_argument("--language", choices=("auto","en","hi"))
    parser.add_argument("--calibrate-distance-mm", type=float, help="Helper-measured total walking distance")
    parser.add_argument("--steps", type=int, help="Counted individual steps for measured-walk calibration")
    parser.add_argument("--self-test", action="store_true", help="Run software tests; no weights/hardware needed")
    args = parser.parse_args()
    if args.detector:
        apply_profile(CONFIG.vision, args.detector)
    if args.language:
        CONFIG.response_language = args.language
    if args.self_test:
        import unittest
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(Path(__file__).parent/"tests")))
        return 0 if result.wasSuccessful() else 1
    if args.calibrate_distance_mm is not None or args.steps is not None:
        if args.calibrate_distance_mm is None or args.steps is None:
            parser.error("Supply both --calibrate-distance-mm and --steps")
        from awareness.distance import Calibration
        calibration = Calibration(CONFIG.distance.calibration_file, CONFIG.distance.default_step_mm)
        calibration.from_walk(args.calibrate_distance_mm, args.steps)
        print(f"Saved step length: {calibration.step_mm:g} mm")
        return 0
    from awareness.common import configure_logging, install_offline_guard
    configure_logging(CONFIG.log_dir)
    if CONFIG.block_internet:
        install_offline_guard()
    if args.list_audio:
        if CONFIG.voice.backend == "inmp441_i2s":
            from hardware.i2s_common import find_i2s_device
            try:
                print("INMP441/MAX98357A shared I2S PCM:", find_i2s_device())
            except Exception as exc:
                print("I2S audio card unavailable:", exc)
                return 1
        else:
            import sounddevice as sd
            print(sd.query_devices())
        return 0
    if args.doctor:
        from awareness.diagnostics import doctor
        return doctor(CONFIG, deep=args.deep, hardware=args.hardware)
    from awareness.app import Application
    from awareness.commands import parse_command
    app = None
    try:
        app = Application(CONFIG, image=args.image, no_speech=args.no_speech)
        signal.signal(signal.SIGTERM, lambda *_: (app.done.set(), app.cancel.set(), app.speaker.stop()))
        app.start(voice=not (args.text or args.once))
        if args.once:
            command = parse_command(args.once)
            if not command:
                parser.error("Unknown command. Try scan, color, read, distance, or help.")
            app.process(command)
            app.speaker.wait_idle()
        elif args.text:
            print("Type commands; enter quit to exit. All inference is local.")
            while not app.done.is_set():
                text = input("> ").strip()
                if text.lower() in ("quit", "exit"):
                    break
                command = parse_command(text)
                if command:
                    app.process(command)
                else:
                    print("Unknown command; type help.")
        else:
            app.run()
    except (KeyboardInterrupt, EOFError):
        return 0
    except Exception as exc:
        import logging
        logging.getLogger(__name__).exception("Startup/runtime failure: %s", exc)
        print("Run .venv/bin/python server.py --doctor --deep --hardware", file=sys.stderr)
        return 1
    finally:
        if app:
            app.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
