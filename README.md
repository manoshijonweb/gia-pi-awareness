# Gia — hardware bring-up

Raspberry Pi 5 Model B Rev 1.0, Pi OS with Python 3.13.5, `aarch64`.

Hostname `pi` (**not** `raspberrypi` — `raspberrypi.local` will not resolve).

## Hardware

| Part | Bus | Status |
|---|---|---|
| VL53L0X time-of-flight ranger | I2C, 0x29 | verified — model `0xEE`, rev `0x10` |
| INMP441 MEMS microphone | I2S capture | verified — speech transcribed back as text |
| MAX98357A class-D amp | I2S playback | verified — 440Hz tone audible |

### Wiring

Both audio parts share BCLK and LRCLK; only the data line differs.

| Pin (BCM) | Header | VL53L0X | INMP441 | MAX98357A |
|---|---|---|---|---|
| GPIO2 | 3 | SDA | — | — |
| GPIO3 | 5 | SCL | — | — |
| GPIO18 | 12 | — | SCK | BCLK |
| GPIO19 | 35 | — | WS | LRC |
| GPIO20 | 38 | — | SD | — |
| GPIO21 | 40 | — | — | DIN |
| 3V3 | 1 / 17 | VIN | VDD | — |
| 5V | 2 / 4 | — | — | VIN |
| GND | 6 / 9 / 39 | GND | GND, L/R | GND, GND |

Three details that are easy to get wrong:

- **INMP441 `L/R` to GND.** Selects the left slot. Floating gives silence.
- **MAX98357A `VIN` to 5V**, not 3.3V. 3.3V works but is very quiet.
- **MAX98357A `SD` left floating.** Averages L+R to mono.

VL53L0X `XSHUT` floats or goes high — never low, or the chip sits in reset and
the bus scans clean.

## Boot configuration

`/boot/firmware/config.txt` (note: `/boot/firmware/`, not `/boot/`, on Pi 5).

```
dtparam=i2c_arm=on

[all]
# Gia: INMP441 (capture) + MAX98357A (playback) on one I2S bus
dtoverlay=googlevoicehat-soundcard
```

Backup of the pre-change file: `/boot/firmware/config.txt.bak-gia`.

`googlevoicehat-soundcard` is the overlay to use despite the name — it is the
generic full-duplex I2S card, and it is what puts capture *and* playback on a
single card. `hifiberry-dac` would drive the MAX98357A but leave no microphone.

`dtoverlay=nospi10` under `[pi5]` is stock Pi OS. Leave it.

After reboot, `aplay -l` and `arecord -l` both show:

```
card 2: sndrpigooglevoi [snd_rpi_googlevoicehat_soundcar]
```

Cards 0 and 1 are the Pi 5's two HDMI outputs and are unrelated.

## Scripts

One file per device. Each is a small module with a single class, and each runs
on its own as a demo. Run one to test that device; import it to build on it.

| File | Class | What it does |
|---|---|---|
| `vl53l0x.py` | `Ranger` | Distance sensing |
| `inmp441.py` | `Microphone` | Recording |
| `max98357a.py` | `Speaker` | Playback |
| `transcribe.py` | `SpeechToText` | Offline speech to text |
| `example.py` | — | All four working together |

### The whole API

```python
from vl53l0x import Ranger
from inmp441 import Microphone
from max98357a import Speaker
from transcribe import SpeechToText

ranger = Ranger()
ranger.distance()                  # mm, or None if nothing is there
ranger.wait_until_closer(600)      # blocks until someone walks up
ranger.wait_until_clear(800)       # blocks until they leave

mic = Microphone()
mic.record(5, "clip.wav")          # save a wav
mic.samples(5)                     # numpy array, DC offset removed
mic.speech_pcm(5)                  # bytes for a recogniser
mic.level()                        # 0.0 to 1.0, how loud right now

speaker = Speaker()
speaker.beep()                     # short cue tone
speaker.tone(440, 2)               # hz, seconds
speaker.play("clip.wav")

ears = SpeechToText()
ears.listen(mic, seconds=5)        # -> "what they said"
ears.transcribe(pcm)               # if you already have audio
```

### Running them directly

```
python3 vl53l0x.py       # live distance readout
python3 inmp441.py       # live level meter
python3 max98357a.py     # test tone
python3 transcribe.py    # beep, listen 5s, print what you said
python3 example.py       # wait for approach -> beep -> listen -> respond
```

### Building on them

`example.py` is the starting point — it waits for someone to approach, beeps,
listens, and responds, in about twenty lines. Replace its `respond()` function
and leave the rest.

The device quirks are handled inside the classes so nothing downstream has to
know about them: the mic's DC offset and 24-in-32-bit framing, the ranger's
out-of-range sentinel, click-free fades on generated tones. Both audio files
find the sound card **by name** from `/proc/asound/cards` rather than
hardcoding `plughw:2,0`, because that index shifts when an HDMI output comes
or goes.

`find_device()` is deliberately duplicated in `inmp441.py` and `max98357a.py`
rather than shared. Six lines of repetition keeps each file something you can
copy into another project on its own.

### Bring-up script

| Script | Does |
|---|---|
| `hwcheck.py` | Platform, I2C bus, 0x29 chip ID, ALSA card inventory |

```
pip install smbus2 adafruit-circuitpython-vl53l0x --break-system-packages
python3 hwcheck.py
python3 range_test.py
python3 audio_test.py
speaker-test -D plughw:2,0 -c 2 -t sine -f 440
```

## Known-good baselines

VL53L0X at ~11cm from a flat target:

```
109 mm, ±2 mm noise, tracking smoothly to 88 mm as the target approached
```

INMP441, 5s of normal room speech:

```
left    DC offset +2424158 (+0.11% FS)   AC rms 54823213 (2.55% FS)   LIVE
right   silent
```

Right channel silent is **correct** — `L/R` tied to GND puts the mic in the
left slot only. Record in stereo, use the left channel.

## Gotchas hit during bring-up

- The INMP441 is 24-bit in 32-bit frames. Record `S32_LE`. `S16_LE` gives noise
  or silence.
- It carries a large DC offset by design. High-pass it in software; it is not a
  wiring fault.
- VL53L0X clone boards ship with a near-invisible protective film over the lens.
  It reads "out of range" forever until peeled.
- Pi 5 `config.txt` lives at `/boot/firmware/config.txt`.
- Passing inline Python to `ssh` from PowerShell mangles the quoting. Write a
  file and `scp` it.
- Speaking directly into the INMP441 clips at 100% FS and wrecks ASR accuracy.
  Normal speech at 15–20cm lands around 15% FS, which is the target.
- `arecord -d` accepts whole seconds only. For fractional windows use `-s`
  (samples per channel) instead.
- Remote level-tests are worthless without a cue the user can perceive — the
  recording starts before they can read an instruction. `transcribe.py` plays a
  beep through the MAX98357A for exactly this reason.

## Speech to text

Vosk, offline, no API key.

```
pip install vosk --break-system-packages
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
python3 transcribe.py --seconds 10
```

`vosk-model-small-en-us-0.15` (40MB, unpacks to 68MB) is fast but lossy — it
fills gaps with plausible English. `vosk-model-en-us-0.22` (~1.8GB) is far more
accurate and the Pi 5 can run it, more slowly.

## Where this stopped — 2026-09-22

Hardware bring-up is **complete**. All three parts verified independently on
real hardware, not just at the signal level:

- VL53L0X read 109mm ±2mm and tracked a target to 88mm
- INMP441 speech came back as readable text through Vosk
- MAX98357A played an audible 440Hz tone, and loopback playback of a live
  recording was audible

The ranger and the audio share no pins and sit on different buses, so
integration from here is a software problem, not a wiring one.

### Open issues

1. **The mic clips.** Every capture peaks at ~100% FS. Not a wiring fault, but
   flat-topped audio badly degrades speech recognition — the garbled Vosk
   transcript was partly this. Find whether it is proximity, a loud room, or
   something near the board, before building anything speech-driven.
2. **The small Vosk model is lossy.** It fills gaps with plausible English.
   `vosk-model-en-us-0.22` (~1.8GB) is far more accurate if accuracy matters.
3. ~~**No version control yet.**~~ Now on GitHub (2026-09-26).

### What Gia actually is — still undecided

A ToF ranger plus speech-in and audio-out is the shape of a machine that
notices someone approach and talks to them. The obvious next pieces would be
wake-on-approach (distance threshold starts a listen), a listening loop that
is not a fixed-length window, and the response path. That is a guess, not a
decision — the purpose has not been written down anywhere yet, and it should
be before the architecture hardens around an assumption.

### Resuming

```
ssh <user>@<pi-address>
cd ~
python3 hwcheck.py             # confirms all three are still live
```
