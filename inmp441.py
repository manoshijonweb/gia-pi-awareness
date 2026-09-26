#!/usr/bin/env python3
"""Microphone (INMP441).

    from inmp441 import Microphone

    mic = Microphone()
    mic.record(5, "clip.wav")    # save 5 seconds to a file
    audio = mic.samples(5)       # numpy array, ready to process
    pcm = mic.speech_pcm(5)      # bytes for a speech recogniser
    peak = mic.level()           # 0.0 to 1.0, how loud it is right now

Run this file directly for a live level meter.

Wiring:  SCK -> pin 12,  WS -> pin 35,  SD -> pin 38,  VDD -> 3V3,  GND -> GND
         L/R -> GND. This is required. Floating, the mic sends silence.

Two quirks of this part, both handled for you below:
  * It sends 24-bit audio inside 32-bit frames, so it must be captured as
    S32_LE. Asking ALSA for S16_LE gives noise.
  * It sits on a big DC offset that changes every time recording starts. That
    is normal. `samples()` removes it; the raw wav from `record()` still has it.
"""

import subprocess
import wave

import numpy as np

RATE = 48000          # the mic's native sample rate
CHANNELS = 2          # it only fills the left one, but ALSA wants a stereo stream
FULL_SCALE = 2 ** 31  # largest possible 32-bit sample


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


class Microphone:
    """The INMP441 on the I2S bus."""

    def __init__(self, device=None):
        self.device = device or find_device()

    def record(self, seconds, path):
        """Record to a wav file and return its path.

        The file keeps the mic's raw format, DC offset and all.
        """
        subprocess.run(
            ["arecord", "-D", self.device,
             "-f", "S32_LE", "-r", str(RATE), "-c", str(CHANNELS),
             # -s counts samples, unlike -d which only accepts whole seconds
             "-s", str(int(seconds * RATE)), path],
            check=True, capture_output=True,
        )
        return path

    def samples(self, seconds):
        """Record and return mono audio as a numpy array, DC offset removed.

        Values are in raw 32-bit units. Divide by FULL_SCALE for -1.0 to 1.0.
        """
        path = self.record(seconds, "/tmp/gia_mic.wav")
        with wave.open(path) as w:
            raw = w.readframes(w.getnframes())

        both_channels = np.frombuffer(raw, dtype="<i4")
        left = both_channels[0::CHANNELS].astype(float)
        return left - left.mean() if left.size else left

    def speech_pcm(self, seconds, rate=16000):
        """Record audio as 16-bit PCM bytes, ready for a speech recogniser.

        Speech recognisers want 16kHz mono. We average every 3 samples, which
        both downsamples 48kHz to 16kHz and smooths out the high frequencies
        that would otherwise alias.
        """
        audio = self.samples(seconds)
        if audio.size == 0:
            return b""

        step = RATE // rate
        usable = (audio.size // step) * step
        audio = audio[:usable].reshape(-1, step).mean(axis=1)

        loudest = np.abs(audio).max()
        if loudest == 0:
            return b""

        # Recognisers expect normal speech levels; this mic runs quiet.
        return (audio / loudest * 0.7 * 32767).astype("<i2").tobytes()

    def level(self, seconds=0.5):
        """How loud it is right now, from 0.0 (silence) to 1.0 (clipping)."""
        audio = self.samples(seconds)
        if audio.size == 0:
            return 0.0
        return float(np.abs(audio).max()) / FULL_SCALE


if __name__ == "__main__":
    mic = Microphone()
    print(f"Level meter on {mic.device}. Ctrl-C to stop.")
    try:
        while True:
            peak = mic.level()
            bar = "#" * int(min(peak, 1.0) * 50)
            warning = "  TOO LOUD" if peak > 0.9 else ""
            print(f"  {peak * 100:5.1f}% |{bar:<50}|{warning}")
    except KeyboardInterrupt:
        print("stopped")
