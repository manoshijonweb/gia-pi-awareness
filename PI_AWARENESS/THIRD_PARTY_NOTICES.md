# Third-party notices and provenance

The application source in this repository is offered under the included MIT license. Third-party software, models,
OS packages and hardware libraries keep their own licenses. No model weights or OS image are bundled in this ZIP.

| Component | Upstream license / note |
|---|---|
| YOLO26m / optional YOLO26s/n | Review Ultralytics upstream AGPL-3.0/Enterprise terms and the ONNX conversion provenance before redistribution |
| PaddleOCR / RapidOCR models | Apache-2.0 project/model provenance; keep exact downloaded notices |
| Vosk API and selected English/Hindi models | Apache-2.0 labels in the Vosk ecosystem/model catalogue |
| ONNX Runtime | MIT |
| Adafruit CircuitPython VL53L0X | MIT |
| Adafruit Blinka dependencies | Keep exact installed package licenses |
| sounddevice | MIT |
| smbus2 | MIT |
| NumPy | BSD family |
| OpenCV / Picamera2 / libcamera / ALSA | Keep the installed distribution's notices |
| eSpeak NG | GPL; redistribution obligations apply |

The hardware adapter structure is based on the four user-supplied hardware helpers for INMP441, MAX98357A,
VL53L0X and hardware sanity checking. The supplied original hardware check is retained for traceability.

This file is a provenance inventory, not legal advice. Review current licenses for every binary/model included in a distributed image.
