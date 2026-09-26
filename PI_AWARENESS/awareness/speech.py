"""Offline eSpeak NG -> ALSA. No shell, network TTS, or audio files.
Half-duplex during playback prevents the device hearing itself as a command.
Long reading has listening gaps for spoken STOP; this is not full-duplex AEC.
"""
from __future__ import annotations
from dataclasses import dataclass
import logging
import queue
import re
import shutil
import subprocess
import threading
import time
from .common import FeatureError


def text_chunks(text: str, maximum: int = 150) -> list[str]:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    chunks, current = [], ""
    for word in text.split():
        if current and len(current) + len(word) + 1 > maximum:
            chunks.append(current)
            current = ""
        current += (" " if current else "") + word
    if current:
        chunks.append(current)
    return chunks


def language_runs(text: str, default: str) -> list[tuple[str, str]]:
    runs = []
    current, language = [], default
    for word in text.split():
        has_hi = any("\u0900" <= ch <= "\u097f" for ch in word)
        has_en = bool(re.search(r"[A-Za-z]", word))
        target = "hi" if has_hi else ("en" if has_en else language)
        if target != language and current:
            runs.append((" ".join(current), language))
            current = []
        language = target
        current.append(word)
    if current:
        runs.append((" ".join(current), language))
    return runs

@dataclass(frozen=True)
class SpeechJob:
    text: str
    language: str
    epoch: int
    command_ended: float | None


def resolve_output_device(configured: str | None) -> str | None:
    """Resolve the assembled MAX98357A output without hard-coding ALSA card numbers."""
    if configured == "i2s-auto":
        from hardware.i2s_common import find_i2s_device
        return find_i2s_device()
    return configured

def resolve_voices(cfg) -> dict[str, str]:
    """Never let an unavailable en-in voice silently eliminate spoken output."""
    data = subprocess.check_output(["espeak-ng", "--voices"], text=True, timeout=10)
    available = {row.split()[1] for row in data.splitlines()[1:] if len(row.split()) > 1}
    english = cfg.english_voice
    if english not in available:
        english = next((v for v in ("en", "en-gb", "en-us") if v in available), None)
        if english is None:
            raise FeatureError("No English eSpeak NG voice installed")
        logging.getLogger(__name__).warning("TTS voice %s absent; using %s", cfg.english_voice, english)
    if cfg.hindi_voice not in available:
        raise FeatureError("Hindi eSpeak NG voice is missing; install the complete espeak-ng package")
    return {"en": english, "hi": cfg.hindi_voice}

class Speaker:
    def __init__(self, cfg, silent: bool = False):
        self.cfg, self.silent = cfg, silent
        self.output_device = None if silent else resolve_output_device(cfg.speaker_device)
        if not silent and (not shutil.which("espeak-ng") or not shutil.which("aplay")):
            raise FeatureError("Install espeak-ng and alsa-utils using bash install.sh")
        self.voices = {"en": cfg.english_voice, "hi": cfg.hindi_voice} if silent else resolve_voices(cfg)
        self.queue = queue.Queue(maxsize=4)
        self.busy = threading.Event()
        self.active = threading.Event()
        self.user_speaking = threading.Event()
        self.done = threading.Event()
        self.lock = threading.RLock()
        self.processes = []
        self.epoch = 0
        self.suppressed_until = 0.0
        self.thread = threading.Thread(target=self._run, name="speaker", daemon=True)
        self.thread.start()

    def input_blocked(self) -> bool:
        return self.busy.is_set() or time.monotonic() < self.suppressed_until

    def say(self, text: str, language: str = "en", command_ended: float | None = None):
        if not text.strip():
            return
        with self.lock:
            job = SpeechJob(text, language, self.epoch, command_ended)
        try:
            self.queue.put_nowait(job)
        except queue.Full:
            logging.getLogger(__name__).warning("Speech queue full; response dropped")

    def stop(self):
        with self.lock:
            self.epoch += 1
            while True:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break
            for process in self.processes:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
            self.suppressed_until = time.monotonic() + self.cfg.echo_tail_s

    def _play(self, text: str, language: str, epoch: int):
        voice = self.voices[language]
        args = ["espeak-ng", "--stdout", "-v", voice, "-s", str(self.cfg.speech_rate), "--stdin"]
        output = ["aplay", "-q"]
        if self.output_device:
            output += ["-D", self.output_device]
        with self.lock:
            if epoch != self.epoch or self.done.is_set():
                return
            synth = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            try:
                play = subprocess.Popen(output, stdin=synth.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            except Exception:
                synth.terminate()
                synth.wait(timeout=2)
                raise
            self.processes = [synth, play]
        synth.stdout.close()  # aplay owns the read end; don't hold a second reader
        try:
            try:
                synth.stdin.write(text.encode("utf-8"))
                synth.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            synth.wait(timeout=max(15, len(text) / 4))
            _, error = play.communicate(timeout=max(15, len(text) / 4))
            if epoch == self.epoch and (synth.returncode != 0 or play.returncode != 0):
                raise FeatureError("Local speech playback failed: " + error.decode("utf-8", errors="replace")[:200])
        finally:
            for process in (synth, play):
                if process.poll() is None:
                    process.kill()
                process.wait()
            with self.lock:
                self.processes = []

    def _run(self):
        log = logging.getLogger(__name__)
        while not self.done.is_set():
            try:
                job = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if job.epoch != self.epoch:
                continue
            self.active.set()
            try:
                if self.silent:
                    print(job.text, flush=True)
                    continue
                chunks = text_chunks(job.text, self.cfg.speech_chunk_chars)
                for index, chunk in enumerate(chunks):
                    if job.epoch != self.epoch or self.done.is_set():
                        break
                    self.busy.set()
                    if index == 0 and job.command_ended is not None:
                        log.info("METRIC end_of_speech_to_tts_dispatch_s=%.3f (not acoustic measurement)",
                                 time.monotonic() - job.command_ended)
                    for run, language in language_runs(chunk, job.language):
                        if job.epoch != self.epoch:
                            break
                        self._play(run, language, job.epoch)
                    self.suppressed_until = time.monotonic() + self.cfg.echo_tail_s
                    self.busy.clear()
                    if index + 1 < len(chunks):
                        end = time.monotonic() + self.cfg.listen_gap_s
                        while time.monotonic() < end and job.epoch == self.epoch and not self.done.is_set():
                            self.done.wait(0.03)
                        # Do not start the next chunk over a command already being spoken.
                        end = time.monotonic() + self.cfg.max_utterance_s + self.cfg.silence_s
                        while self.user_speaking.is_set() and time.monotonic() < end and job.epoch == self.epoch:
                            if self.done.wait(0.03):
                                break
            except Exception:
                log.exception("Speech output failed; check speaker/ALSA configuration")
            finally:
                self.busy.clear()
                self.active.clear()

    def wait_idle(self, timeout: float = 180.0):
        end = time.monotonic() + timeout
        # Account for the small interval between queue.get() and active.set().
        time.sleep(0.05)
        while (not self.queue.empty() or self.active.is_set()) and time.monotonic() < end:
            time.sleep(0.05)

    def close(self):
        self.stop()
        self.done.set()
        self.thread.join(timeout=3)
