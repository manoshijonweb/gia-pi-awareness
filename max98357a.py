#!/usr/bin/env python3
"""Speaker (MAX98357A amplifier).

    from max98357a import Speaker

    speaker = Speaker()
    speaker.beep()              # short cue, e.g. "start talking now"
    speaker.tone(440, 2)        # 440Hz for 2 seconds
    speaker.play("clip.wav")    # play a wav file

Run this file directly to play a test tone.

Wiring:  BCLK -> pin 12,  LRC -> pin 35,  DIN -> pin 40,  GND -> GND
         VIN -> 5V. On 3.3V it works but is almost inaudible.
         SD must float or be high. Pulled to GND the amp is switched off.

It shares BCLK and LRC with the microphone -- only the data pin differs, which
is why one overlay covers both.
"""

import math
import struct
import subprocess
import wave

RATE = 44100

# Full volume through a small speaker distorts. This is comfortable.
VOLUME = 0.35


def find_device():
    """Locate the I2S sound card by name and return its ALSA device string.

    Looked up by name rather than hardcoded, because the card number shifts
    when an HDMI monitor is plugged in or unplugged.
    """
    with open("/proc/asound/cards") as f:
        for line in f:
            if "sndrpigooglevoi" in line:
                return f"plughw:{int(line.split()[0])},0"
    raise RuntimeError(
        "I2S sound card not found. Check that config.txt has "
        "dtoverlay=googlevoicehat-soundcard and that the Pi has rebooted."
    )


class Speaker:
    """The MAX98357A on the I2S bus."""

    def __init__(self, device=None):
        self.device = device or find_device()

    def play(self, path):
        """Play a wav file."""
        subprocess.run(
            ["aplay", "-D", self.device, path],
            check=True, capture_output=True,
        )

    def tone(self, hz=440, seconds=1.0, volume=VOLUME):
        """Play a steady tone."""
        self._play_samples(_sine(hz, seconds, volume))

    def beep(self, hz=880, seconds=0.25, volume=VOLUME):
        """Play a short cue tone -- useful for signalling "your turn"."""
        self._play_samples(_sine(hz, seconds, volume, fade=True))

    def _play_samples(self, samples):
        """Write samples to a temporary wav and play it."""
        path = "/tmp/gia_speaker.wav"
        _write_wav(path, samples)
        self.play(path)


def _sine(hz, seconds, volume, fade=False):
    """Generate a sine wave as a list of floats between -1.0 and 1.0."""
    total = int(RATE * seconds)
    out = []
    for i in range(total):
        # Fading in and out stops the click a sudden start makes through a
        # class-D amplifier.
        envelope = (0.5 - 0.5 * math.cos(2 * math.pi * i / total)) if fade else 1.0
        out.append(volume * envelope * math.sin(2 * math.pi * hz * i / RATE))
    return out


def _write_wav(path, samples):
    """Save floats between -1.0 and 1.0 as a 16-bit stereo wav."""
    frames = []
    for value in samples:
        clamped = max(-1.0, min(1.0, value))
        sample = int(clamped * 32767)
        frames.append(struct.pack("<hh", sample, sample))

    with wave.open(path, "w") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(b"".join(frames))


if __name__ == "__main__":
    speaker = Speaker()
    print(f"Playing a test tone on {speaker.device}")
    speaker.tone(440, 2)
    print("Heard nothing? Check VIN is on 5V (not 3.3V), DIN is on pin 40,")
    print("and the SD pin is not pulled to GND.")
