#!/usr/bin/env python3
"""MAX98357A I2S amplifier diagnostic helper.

Wiring from the supplied hardware test:
  BCLK -> physical pin 12 (GPIO18)
  LRC  -> physical pin 35 (GPIO19)
  DIN  -> physical pin 40 (GPIO21)
  VIN  -> 5V
  GND  -> GND
  SD   -> floating or high

The amplifier shares BCLK/LRCLK with the INMP441 and uses the playback side of
the same googlevoicehat ALSA card.
"""
from __future__ import annotations
import math
import struct
import subprocess
import wave
from .i2s_common import find_i2s_device

RATE = 44100
VOLUME = 0.35


class Speaker:
    def __init__(self, device: str | None = None):
        self.device = device or find_i2s_device()

    def play(self, path: str):
        subprocess.run(["aplay", "-q", "-D", self.device, path], check=True)

    def tone(self, hz: float = 440, seconds: float = 1.0, volume: float = VOLUME):
        self._play_samples(_sine(hz, seconds, volume))

    def beep(self, hz: float = 880, seconds: float = 0.25, volume: float = VOLUME):
        self._play_samples(_sine(hz, seconds, volume, fade=True))

    def _play_samples(self, samples):
        path = "/tmp/pi_awareness_speaker.wav"
        _write_wav(path, samples)
        self.play(path)


def _sine(hz, seconds, volume, fade=False):
    total = max(1, int(RATE * seconds))
    out = []
    for i in range(total):
        envelope = (0.5 - 0.5 * math.cos(2 * math.pi * i / total)) if fade else 1.0
        out.append(volume * envelope * math.sin(2 * math.pi * hz * i / RATE))
    return out


def _write_wav(path, samples):
    frames = []
    for value in samples:
        clamped = max(-1.0, min(1.0, value))
        sample = int(clamped * 32767)
        frames.append(struct.pack("<hh", sample, sample))
    with wave.open(path, "w") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(b"".join(frames))


if __name__ == "__main__":
    speaker = Speaker()
    print(f"Playing test tone on {speaker.device}")
    speaker.tone(440, 2)
