# Hardware — assembled Pi 5 profile

This release is not a generic pinout. It implements the hardware supplied for this build.

## I2S audio

Both audio devices share GPIO18/GPIO19 clocks and use different data directions:

```text
Pi 5 GPIO18 / pin 12  -> INMP441 SCK  + MAX98357A BCLK
Pi 5 GPIO19 / pin 35  -> INMP441 WS   + MAX98357A LRC
Pi 5 GPIO20 / pin 38  <- INMP441 SD
Pi 5 GPIO21 / pin 40  -> MAX98357A DIN
3V3                    -> INMP441 VDD
5V                     -> MAX98357A VIN
GND                    -> both grounds
INMP441 L/R            -> GND
MAX98357A SD            -> float/high
```

The installer adds:

```text
dtparam=i2s=on
dtoverlay=googlevoicehat-soundcard
```

The runtime locates `sndrpigooglevoi` in `/proc/asound/cards` and uses `plughw:<card>,0`.
This prevents HDMI insertion/removal from breaking a hard-coded ALSA card number.

INMP441 capture is S32_LE, 48 kHz, 2-channel framed; only the left channel carries the mic because L/R is tied low.

## VL53L0X

```text
pin 3  / SDA -> SDA
pin 5  / SCL -> SCL
3V3          -> VIN
GND          -> GND
XSHUT        -> float/high
```

I2C address: `0x29`. The supplied helper uses a 200,000 µs measurement timing budget and treats 8190 as no-return.
The service does not poll the sensor while idle; the `distance` command opens it on demand.

## Camera Rev 1.3

Use the correct Pi 5 camera ribbon/cable orientation. The software uses Picamera2 and camera index 0.
Camera Rev 1.3 / Camera Module 1 is fixed-focus, so autofocus is disabled in `settings.py`.
The camera is not streamed in idle mode.

## Power

Software idle mode eliminates camera streaming, detector/OCR inference and ToF polling. The microphone, ALSA capture,
VAD and resident speech recognizers remain active because the device must hear commands. This is not deep sleep.
