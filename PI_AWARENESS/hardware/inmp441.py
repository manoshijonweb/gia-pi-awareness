#!/usr/bin/env python3
"""INMP441 microphone adapter for the assembled Pi 5.

Wiring from the supplied hardware test:
  SCK/BCLK -> physical pin 12 (GPIO18)
  WS/LRCLK -> physical pin 35 (GPIO19)
  SD/DOUT  -> physical pin 38 (GPIO20)
  VDD      -> 3V3
  GND      -> GND
  L/R      -> GND (left channel)

The microphone supplies 24-bit data in 32-bit I2S frames.  We therefore keep
ALSA at S32_LE/48 kHz stereo and perform the small 48 -> 16 kHz conversion in
Python.  Asking this wiring/overlay for S16_LE directly can produce unusable
capture on this setup.
"""
from __future__ import annotations
import subprocess
import threading
import time
import wave
from pathlib import Path
import numpy as np
from .i2s_common import find_i2s_device

RATE = 48000
CHANNELS = 2
CHANNEL_INDEX = 0
FULL_SCALE = float(2**31)


def _read_exact(stream, count: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < count:
        part = stream.read(count - len(chunks))
        if not part:
            break
        chunks.extend(part)
    return bytes(chunks)


def convert_i2s_block(raw: bytes, *, channel_index: int = CHANNEL_INDEX,
                      input_rate: int = RATE, output_rate: int = 16000,
                      gain: float = 10.0) -> bytes:
    """Convert one S32_LE stereo block to mono signed-16 PCM.

    A per-block DC removal mirrors the behaviour of the supplied standalone
    microphone helper.  A 3-sample box average provides a lightweight speech
    low-pass while converting 48 kHz to 16 kHz.  Gain is fixed/configurable;
    we intentionally do not normalize every silent block because that would
    amplify background noise into false voice activity.
    """
    if not raw:
        return b""
    data = np.frombuffer(raw, dtype="<i4")
    usable = (data.size // CHANNELS) * CHANNELS
    if usable == 0:
        return b""
    stereo = data[:usable].reshape(-1, CHANNELS)
    if channel_index < 0 or channel_index >= CHANNELS:
        raise ValueError("INMP441 channel index must be 0 or 1")
    mono = stereo[:, channel_index].astype(np.float64)
    if mono.size == 0:
        return b""
    mono -= mono.mean()
    if input_rate % output_rate:
        raise ValueError("INMP441 lightweight converter requires an integer rate ratio")
    step = input_rate // output_rate
    usable = (mono.size // step) * step
    if usable == 0:
        return b""
    mono = mono[:usable].reshape(-1, step).mean(axis=1)
    scaled = np.clip((mono / FULL_SCALE) * gain * 32767.0, -32768, 32767)
    return scaled.astype("<i2").tobytes()


class INMP441Stream:
    """Persistent low-overhead ALSA capture used by the always-listening VAD."""
    def __init__(self, *, device: str | None = None, block_ms: int = 50,
                 output_rate: int = 16000, channel_index: int = 0,
                 gain: float = 10.0, on_block=None, on_error=None):
        self.device = device or find_i2s_device()
        self.block_ms = int(block_ms)
        self.output_rate = int(output_rate)
        self.channel_index = int(channel_index)
        self.gain = float(gain)
        self.on_block = on_block
        self.on_error = on_error
        self.process: subprocess.Popen | None = None
        self.thread: threading.Thread | None = None
        self.done = threading.Event()
        self.error = ""
        frames = max(1, int(RATE * self.block_ms / 1000))
        self.raw_block_bytes = frames * CHANNELS * 4

    def start(self):
        if self.process is not None:
            return self
        cmd = [
            "arecord", "-q", "-D", self.device,
            "-f", "S32_LE", "-r", str(RATE), "-c", str(CHANNELS),
            "-t", "raw",
        ]
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
        )
        self.thread = threading.Thread(target=self._run, name="inmp441", daemon=True)
        self.thread.start()
        return self

    def _run(self):
        assert self.process is not None and self.process.stdout is not None
        try:
            while not self.done.is_set():
                raw = _read_exact(self.process.stdout, self.raw_block_bytes)
                if len(raw) != self.raw_block_bytes:
                    if self.done.is_set():
                        break
                    stderr = b""
                    if self.process.stderr is not None:
                        try:
                            stderr = self.process.stderr.read(512)
                        except Exception:
                            pass
                    raise RuntimeError(
                        "INMP441 arecord stopped unexpectedly: "
                        + stderr.decode("utf-8", errors="replace").strip()
                    )
                pcm = convert_i2s_block(
                    raw, channel_index=self.channel_index,
                    output_rate=self.output_rate, gain=self.gain,
                )
                if pcm and self.on_block:
                    self.on_block(time.monotonic(), pcm)
        except Exception as exc:
            self.error = str(exc)
            if self.on_error and not self.done.is_set():
                self.on_error(exc)
        finally:
            self._terminate()

    def _terminate(self):
        proc = self.process
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=1.5)

    def close(self):
        self.done.set()
        self._terminate()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        self.process = None


class Microphone:
    """Standalone diagnostic API retained from the supplied hardware helper."""
    def __init__(self, device: str | None = None):
        self.device = device or find_i2s_device()

    def record(self, seconds: float, path: str):
        subprocess.run([
            "arecord", "-q", "-D", self.device,
            "-f", "S32_LE", "-r", str(RATE), "-c", str(CHANNELS),
            "-d", str(max(1, int(round(seconds)))), path,
        ], check=True)
        return path

    def samples(self, seconds: float):
        path = "/tmp/pi_awareness_mic.wav"
        self.record(seconds, path)
        with wave.open(path) as wav:
            raw = wav.readframes(wav.getnframes())
        both = np.frombuffer(raw, dtype="<i4")
        left = both[CHANNEL_INDEX::CHANNELS].astype(np.float64)
        return left - left.mean() if left.size else left

    def speech_pcm(self, seconds: float, rate: int = 16000, gain: float = 10.0):
        path = "/tmp/pi_awareness_mic.wav"
        self.record(seconds, path)
        with wave.open(path) as wav:
            raw = wav.readframes(wav.getnframes())
        return convert_i2s_block(raw, output_rate=rate, gain=gain)

    def level(self, seconds: float = 0.5):
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
            print(f"  {peak * 100:5.1f}% |{bar:<50}|")
    except KeyboardInterrupt:
        pass
