# Sources and provenance

Primary external references used by this release:

- Raspberry Pi camera software / Picamera2:
  https://www.raspberrypi.com/documentation/computers/camera_software.html
- Raspberry Pi camera modules, including Camera Module 1:
  https://www.raspberrypi.com/documentation/accessories/camera.html
- Raspberry Pi Device Tree overlays and `googlevoicehat-soundcard`:
  https://github.com/raspberrypi/firmware/blob/master/boot/overlays/README
- Adafruit CircuitPython VL53L0X package:
  https://pypi.org/project/adafruit-circuitpython-vl53l0x/
- Ultralytics detection documentation:
  https://docs.ultralytics.com/tasks/detect/
- Ultralytics Raspberry Pi guide:
  https://docs.ultralytics.com/guides/raspberry-pi/
- ONNX Community YOLO26m conversion:
  https://huggingface.co/onnx-community/yolo26m-ONNX
- Vosk model catalogue:
  https://alphacephei.com/vosk/models
- RapidOCR model catalogue:
  https://rapidai.github.io/RapidOCRDocs/main/model_list/

The four hardware helper files supplied with this task are the basis for the assembled pinout and low-level device contracts.
The original `hwcheck.py` is retained as `hardware/hwcheck_original.py` for traceability.
