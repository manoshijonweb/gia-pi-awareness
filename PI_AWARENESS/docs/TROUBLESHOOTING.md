# Troubleshooting

## Service keeps restarting

```bash
sudo systemctl status pi-awareness
journalctl -u pi-awareness -b -n 200 --no-pager
```

The service intentionally exits/restarts if the voice listener dies.

## I2S card missing

```bash
cat /proc/asound/cards
arecord -l
aplay -l
grep -E 'i2s|googlevoicehat' /boot/firmware/config.txt
```

Expected card name contains `sndrpigooglevoi`. Confirm `dtoverlay=googlevoicehat-soundcard`, then reboot.

## INMP441 is silence

Confirm L/R is tied to GND, VDD is 3.3 V, and GPIO18/19/20 are connected as documented. Stop the service and run:

```bash
.venv/bin/python tools/mic_test.py --seconds 15
```

The software deliberately captures S32_LE/48 kHz. Do not replace that path with direct S16_LE capture for this wiring.

## Speaker silent

MAX98357A VIN should be 5 V, DIN GPIO21/pin 40, and SD must not be held low. Stop the service and run:

```bash
.venv/bin/python tools/speaker_test.py
python -m hardware.max98357a
```

## Camera unavailable

```bash
rpicam-hello --list-cameras
```

If available on your OS, this should show the attached camera. The code uses Picamera2; it does not use the deprecated legacy Picamera stack.
Check the Pi 5 ribbon cable and connector orientation.

## VL53L0X unavailable

```bash
i2cdetect -y 1
```

Expected address is `29`. Confirm SDA/SCL, 3.3 V, ground, and that XSHUT is not low.

## `Device or resource busy`

The `pi-awareness` service normally owns the microphone. Stop it before manual capture or hardware tests:

```bash
sudo systemctl stop pi-awareness
# run test
sudo systemctl start pi-awareness
```
