# Validation report — PI_AWARENESS 1.2.0 hardware-profile release

Build date: 22 September 2026.

This source release is configured for the user's assembled Raspberry Pi 5 hardware profile:
Raspberry Pi Camera Rev 1.3, INMP441 I2S microphone, MAX98357A I2S amplifier, and VL53L0X.
Model weights are intentionally not bundled; `install.sh` provisions them once.

## Actually executed in the build environment

Environment: Linux x86_64, Python 3.13.5, NumPy 2.3.5, OpenCV 4.13.0.
This machine is **not a Raspberry Pi** and cannot electrically validate the attached hardware.

| Check | Result |
|---|---|
| `python3 server.py --self-test` | **153 tests passed**, 0 failures/errors |
| Compile all Python sources | Passed |
| `bash -n install.sh run.sh tools/install_service.sh` | Passed |
| Exact boot phrase unit test | Passed: `I am ready` is queued before voice capture starts |
| Object-count wording | Passed: repeated detections are counted, including `2 people and 1 chair` |
| On-demand camera path | Passed with mocked Picamera2 backend; close verified |
| INMP441 conversion | Passed with synthetic S32_LE/48 kHz/stereo input to 16 kHz mono PCM |
| VL53L0X on-demand/fail-closed logic | Passed with mocked hardware driver |
| Service profile assertions | Passed: auto-enable, restart-on-failure and private network namespace |
| Config overlay assertions | Passed: I2C, I2S, googlevoicehat sound card, camera auto-detect |
| Existing YOLO26/OCR/voice/software tests | Passed |

The tests cover command parsing in English/Hindi, VAD routing, audio gating, calibration,
object-count descriptions, YOLO26 output decoding, OCR preprocessing/decoding, color processing,
camera lifecycle, offline network blocking, model archive/checksum safety, and error/fail-closed paths.

## Deliberately changed from the previous generic build

- INMP441 is the default microphone backend. It captures S32_LE, 48 kHz, two-channel-framed I2S,
  selects the left channel and converts to 16 kHz signed-16 PCM for Vosk.
- MAX98357A playback auto-resolves the shared `sndrpigooglevoi` ALSA card instead of using a fixed card number.
- Raspberry Pi Camera Rev 1.3 uses Picamera2 and is opened only for `scan`, `color`, or `read`.
- VL53L0X replaces the old VL53L1X path and is read only when the `distance` command is requested.
- Silent idle audio performs RMS/VAD first; Vosk decoding begins only after speech is detected.
- `install.sh` installs/enables `pi-awareness.service`; service starts at boot after the required reboot.
- Successful voice startup speaks exactly `I am ready` once, then enables microphone capture.
- `scan` groups detections by COCO class and speaks counts.

## Not executed / not established here

- Fresh Raspberry Pi OS package installation on a physical Pi 5.
- Device Tree overlay activation and reboot on the user's Pi.
- Real INMP441 electrical capture, analog acoustic sensitivity or final VAD threshold.
- Real MAX98357A playback volume/noise and simultaneous I2S capture/playback behaviour.
- Real Raspberry Pi Camera Rev 1.3 framing/focus/exposure or Picamera2 startup time.
- Real VL53L0X accuracy/range on the assembled device.
- Downloading/loading the actual neural model binaries in this build environment.
- Real YOLO26m, OCR, or Vosk accuracy/latency on the Pi.
- End-of-command to first audible response latency, idle wall power, battery runtime, or enclosure thermals.
- Safety certification or suitability as a sole navigation aid.

Software/mocked tests do not establish the physical results above. After installation and reboot,
run the exact hardware checks in `README.md` and `docs/ACCEPTANCE.md`. In particular, tune
`voice.i2s_gain` and `voice.rms_threshold` from measured quiet/speech values rather than guessing.
