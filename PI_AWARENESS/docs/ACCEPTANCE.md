# Acceptance checklist

Run tests on the assembled Raspberry Pi 5, not only on a desktop.

1. Install and reboot. Confirm the service starts and says exactly **"I am ready"** once.
2. Leave it idle for at least 10 minutes. Confirm no camera streaming process and no continuous VL53L0X polling.
3. In a quiet room, confirm microphone noise stays below the configured VAD threshold and normal speech rises above it.
4. Test all English commands: scan, color, read, distance, repeat, stop.
5. Test the Hindi equivalents.
6. Put two people/chairs or repeated COCO objects in view and confirm `scan` announces counts rather than only unique class names.
7. Test 20 varied object scenes and record missed/duplicate detections.
8. Test OCR on English, Hindi and mixed-script samples under normal lighting.
9. Test color under daylight and indoor lighting; it is a descriptive classifier, not a colorimeter.
10. Measure the VL53L0X against known distances in its configured ~30–1200 mm range. Invalid/no-return must never be announced as clear.
11. Calibrate the user's step length and repeat distance tests.
12. Measure full command-to-first-meaningful-audio latency, not just ONNX inference time.
13. Run sustained use while monitoring `vcgencmd measure_temp` and `vcgencmd get_throttled`.
14. Disconnect network access and confirm scan/color/read/distance/voice continue to operate.

This prototype must not be accepted as a sole navigation or hazard-avoidance system.
