# PI_AWARENESS — Raspberry Pi 5 voice-first offline awareness device

Version **1.2.0** · 22 September 2026

This release is built around the **actual assembled hardware**, not generic placeholders:

- Raspberry Pi 5, Raspberry Pi OS Lite 64-bit
- Raspberry Pi Camera Rev 1.3 / Camera Module 1 on CSI, through Picamera2
- INMP441 I2S MEMS microphone
- MAX98357A I2S amplifier/speaker
- VL53L0X ToF distance sensor at I2C address `0x29`
- YOLO26m 640×640 ONNX CPU object detection
- PP-OCRv5 mobile ONNX English + Devanagari OCR
- Vosk offline English/Hindi command recognition
- eSpeak NG offline English/Hindi speech output

Normal runtime uses **no internet**. Installation needs internet once to install packages and download model weights.

> This is an experimental information aid, not a replacement for a white cane, guide dog,
> orientation-and-mobility training, or a human helper. A missed object or failed distance reading
> must never be interpreted as a clear path.

## What happens after boot

`install.sh` installs and **enables `pi-awareness.service` automatically**. After the next reboot,
the service loads its offline speech models and detector, speaks exactly:

> **I am ready**

and then listens for voice commands.

The service is power-aware while idle:

- the INMP441 capture stream stays active so voice commands can be heard;
- only a lightweight RMS/VAD path runs on silent audio;
- Vosk receives audio only after speech crosses the VAD threshold;
- the CSI camera is **not continuously streaming**;
- VL53L0X is **not continuously polled**;
- YOLO/OCR inference runs only when requested;
- YOLO26m is warmed once at boot to avoid first-scan graph startup latency, then consumes no detector CPU while idle.

This is not hardware deep sleep: a device that must hear speech continuously cannot power off the CPU and microphone.
Raspberry Pi OS is left free to perform its normal CPU frequency scaling while the service is idle.

## Voice commands

| English | Hindi | Action |
|---|---|---|
| `scan`, `what is this?`, `what is in front of me?` | `देखो`, `यह क्या है`, `मेरे सामने क्या है` | Capture one frame and announce detected object counts |
| `color`, `what color?` | `रंग`, `रंग बताओ` | Capture one frame and announce the central dominant color |
| `read`, `read text` | `पढ़ो`, `क्या लिखा है` | Capture one frame, OCR it, and speak the text |
| `distance`, `how far?` | `दूरी`, `दूरी बताओ` | Take an on-demand VL53L0X reading and report calibrated steps |
| `repeat` | `दोहराओ` | Repeat the last result |
| `stop` | `रुको` | Stop the current spoken/processing response where supported |
| `calibrate` | `कदम नापो` | Start step-length calibration |
| `speak hindi` / `speak english` | `हिंदी में बोलो` / `अंग्रेजी में बोलो` | Change response language |

### Object counts

`scan` counts detections by COCO class. Example:

```text
I see 2 people, 1 chair, and 1 bottle.
```

The model still recognises the standard 80 COCO categories; it does not identify arbitrary product identities.

## Exact hardware profile

### INMP441 microphone

The supplied hardware helper established this wiring and capture contract:

| INMP441 | Pi 5 |
|---|---|
| SCK / BCLK | physical pin 12 / GPIO18 |
| WS / LRCLK | physical pin 35 / GPIO19 |
| SD / DOUT | physical pin 38 / GPIO20 |
| VDD | 3.3 V |
| GND | GND |
| L/R | **GND** |

The repo captures it as **S32_LE, 48 kHz, 2-channel-framed I2S**, selects the left channel,
removes DC offset, averages 48 kHz → 16 kHz, applies configurable gain, and feeds 16-bit PCM to Vosk.

### MAX98357A speaker amplifier

| MAX98357A | Pi 5 |
|---|---|
| BCLK | physical pin 12 / GPIO18 |
| LRC | physical pin 35 / GPIO19 |
| DIN | physical pin 40 / GPIO21 |
| VIN | 5 V |
| GND | GND |
| SD | float/high |

INMP441 and MAX98357A share BCLK/LRCLK. The install script enables the single
`googlevoicehat-soundcard` overlay used by the supplied hardware test. Card numbers are discovered by
name at runtime; they are not hard-coded.

### VL53L0X

| VL53L0X | Pi 5 |
|---|---|
| SDA | physical pin 3 |
| SCL | physical pin 5 |
| VIN | 3.3 V |
| GND | GND |
| XSHUT | float/high |

The actual supplied VL53L0X helper treats about **30–1200 mm** as the useful configured range.
This is intentionally different from the older repo's VL53L1X/4 m assumption.

### Raspberry Pi Camera Rev 1.3

The repo selects `picamera2`, camera index `0`, 1280×960 capture, and no autofocus. The Rev 1.3 /
Camera Module 1 hardware is fixed-focus. Each vision command opens the camera, obtains a fresh frame,
and closes it again.

## Fresh installation on the Pi

Copy/extract the ZIP onto the Pi and run:

```bash
cd ~/PI_AWARENESS
bash install.sh
sudo reboot
```

`install.sh`:

1. verifies Raspberry Pi 5 + 64-bit OS;
2. installs Pi camera/audio/I2C packages;
3. enables I2C and I2S;
4. adds `dtoverlay=googlevoicehat-soundcard` and `camera_auto_detect=1` if missing;
5. creates `.venv`;
6. installs the pinned runtime dependencies;
7. downloads the detector/OCR/Vosk assets once;
8. runs software/model diagnostics;
9. installs and enables `pi-awareness.service`.

The service is intentionally not started before the reboot because Device Tree and group changes may not yet be active.

## Service operation

```bash
sudo systemctl status pi-awareness
journalctl -u pi-awareness -f
sudo systemctl restart pi-awareness
sudo systemctl disable --now pi-awareness
```

The service has a private network namespace, so the application process cannot use the internet during normal operation.
SSH/networking for the rest of the Pi is not disabled.

## Hardware test after installation

The service owns the I2S microphone. Stop it before exclusive hardware diagnostics:

```bash
cd ~/PI_AWARENESS
sudo systemctl stop pi-awareness

.venv/bin/python tools/hwcheck.py
.venv/bin/python tools/mic_test.py --seconds 15
.venv/bin/python tools/speaker_test.py
.venv/bin/python server.py --doctor --deep --hardware

sudo systemctl start pi-awareness
```

The hardware check tests the shared I2S card, microphone signal, speaker beep, camera frame and VL53L0X response.

## Manual application test

Stop the service first, then:

```bash
sudo systemctl stop pi-awareness
source .venv/bin/activate
python server.py
```

Press `Ctrl+C`, then restart the service:

```bash
sudo systemctl start pi-awareness
```

For typed commands without the microphone:

```bash
python server.py --text --no-speech
```

For a specific local image:

```bash
python server.py --image /path/test.jpg --once scan --no-speech
python server.py --image /path/test.jpg --once read --no-speech
python server.py --image /path/test.jpg --once color --no-speech
```

## Tune the microphone

Default INMP441 conversion settings are:

```python
CONFIG.voice.i2s_gain = 10.0
CONFIG.voice.rms_threshold = 180.0
```

Run `tools/mic_test.py` in a quiet room. Quiet blocks should stay below the threshold and normal speech should rise clearly above it.
Do not lower the threshold until silence/noise values are known.

## Step calibration

With a helper, walk a measured safe distance and count individual steps. Example: 10 steps over 6500 mm:

```bash
.venv/bin/python server.py --calibrate-distance-mm 6500 --steps 10
```

That saves 650 mm/step. The distance sensor measures from the sensor position, not from the user's toes.

## Power/latency trade-off

Default choices intentionally balance idle power and response speed:

- camera: on-demand and closed immediately after capture;
- VL53L0X: on-demand, one approximately 200 ms reading per distance query;
- YOLO26m: graph warmed at service start and retained in RAM;
- OCR: loaded on first `read` command and retained afterward;
- two small Vosk models: resident so English/Hindi commands remain immediately available.

Set `CONFIG.power.preload_detector=False` to minimize boot work/RAM, but the first `scan` will be slower.
A true ultra-low-power standby that wakes the Pi from speech would require separate always-on wake-word hardware or a lower-power coprocessor.

## Repository map

```text
server.py                 entry point
settings.py               assembled-hardware configuration
install.sh                Pi provisioning + service enable
run.sh                    manual launcher
awareness/                voice, camera, vision, OCR, color, distance, TTS
hardware/                 INMP441, MAX98357A, VL53L0X and I2S helpers
hardware/hwcheck_original.py  original supplied hardware sanity-check retained for reference
tools/hwcheck.py          integrated hardware test
tools/install_service.sh  systemd unit installer
tools/mic_test.py         live INMP441 RMS/VAD tuning
tools/speaker_test.py     bilingual speaker test
tools/benchmark.py        feature timing measurements
tests/                    software/mocked-hardware tests
models/                   populated by install.sh
state/                    step calibration / installation state
logs/                     rotating runtime logs
```

See `docs/HARDWARE.md`, `docs/ACCEPTANCE.md`, and `TEST_REPORT.md` before a supervised user trial.
