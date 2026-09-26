"""Low-power offline voice-command listener.

Idle behaviour is deliberately small: ALSA capture blocks in the kernel, Python computes
one RMS value per short block, and Vosk only receives audio after the VAD threshold is
crossed.  Camera, detector, OCR and distance work are not run while waiting for speech.

The Pi hardware profile uses the INMP441 S32_LE/48 kHz stereo-framed stream supplied by
the assembled hardware.  A sounddevice backend remains available for desktop tests.
"""
from __future__ import annotations
from collections import deque
import json
import logging
import queue
import threading
import time
import numpy as np
from .commands import grammar_phrases, parse_command, select_candidate
from .common import FeatureError


def choose_input(cfg):
    """Resolve the desktop/sounddevice backend. The INMP441 backend is fixed at 16 kHz after conversion."""
    if getattr(cfg, "backend", "sounddevice") == "inmp441_i2s":
        return 16000, {"name": "INMP441 via shared I2S ALSA card", "default_samplerate": 48000,
                       "max_input_channels": 2}
    import sounddevice as sd
    info = sd.query_devices(cfg.microphone, "input")
    if cfg.channels < 1 or cfg.channel_index not in range(cfg.channels):
        raise FeatureError("Invalid microphone channels/channel_index")
    if info["max_input_channels"] < cfg.channels:
        raise FeatureError("Configured microphone channel count is unavailable")
    last = None
    for rate in dict.fromkeys([cfg.preferred_sample_rate, int(info["default_samplerate"])]):
        try:
            sd.check_input_settings(device=cfg.microphone, channels=cfg.channels,
                                    samplerate=rate, dtype="int16")
            return int(rate), info
        except Exception as exc:
            last = exc
    raise FeatureError(f"Cannot open microphone at preferred or native sample rate: {last}")


def vocabulary_grammar(model, language: str, require_prefix: bool) -> list[str]:
    phrases = grammar_phrases(language, require_prefix)
    find_word = getattr(model, "vosk_model_find_word", None)
    if find_word:
        phrases = [p for p in phrases if p == "[unk]" or all(find_word(word) >= 0 for word in p.split())]
    if len(phrases) < 2:
        raise FeatureError(f"No command vocabulary found for {language}")
    return phrases


class CommandListener:
    def __init__(self, cfg, speaker, callback, log_text: bool = False):
        import vosk
        self.cfg, self.speaker, self.callback = cfg, speaker, callback
        self.log_text = log_text
        self.rate, info = choose_input(cfg)
        self.queue = queue.Queue(maxsize=24)
        self.done = threading.Event()
        self.models, self.recognizers, self.coverage = {}, {}, {}
        self.dropped = 0
        self.discontinuity = threading.Event()
        self.audio_error = ""
        vosk.SetLogLevel(-1)
        for language in cfg.languages:
            path = cfg.asr_hi if language == "hi" else cfg.asr_en
            if not (path / "am/final.mdl").is_file():
                raise FeatureError(f"Missing local Vosk {language} model: {path}")
            model = vosk.Model(str(path))
            self.models[language] = model
            phrases = vocabulary_grammar(model, language, cfg.require_wake_prefix)
            self.coverage[language] = sorted({command.intent for p in phrases
                if (command := parse_command(p, language, cfg.require_wake_prefix))})
            missing = {"scan", "color", "read", "distance"} - set(self.coverage[language])
            if missing:
                logging.getLogger(__name__).warning("ASR %s vocabulary lacks intents %s", language, missing)
            if cfg.use_grammar:
                try:
                    rec = vosk.KaldiRecognizer(model, self.rate, json.dumps(phrases, ensure_ascii=False))
                except Exception:
                    logging.getLogger(__name__).warning(
                        "Grammar unavailable for %s; using exact-whitelisted full decoder", language)
                    rec = vosk.KaldiRecognizer(model, self.rate)
            else:
                rec = vosk.KaldiRecognizer(model, self.rate)
            rec.SetWords(True)
            self.recognizers[language] = rec
        if not self.recognizers:
            raise FeatureError("Enable at least one ASR language in settings.py")

        backend = getattr(cfg, "backend", "sounddevice")
        if backend == "inmp441_i2s":
            from hardware.inmp441 import INMP441Stream
            self.stream = INMP441Stream(
                block_ms=cfg.block_ms,
                output_rate=self.rate,
                channel_index=cfg.channel_index,
                gain=cfg.i2s_gain,
                on_block=self._capture_i2s,
                on_error=self._audio_failed,
            )
        elif backend == "sounddevice":
            import sounddevice as sd
            self.stream = sd.RawInputStream(
                samplerate=self.rate,
                blocksize=int(self.rate * cfg.block_ms / 1000),
                device=cfg.microphone,
                channels=cfg.channels,
                dtype="int16",
                callback=self._capture,
            )
        else:
            raise FeatureError(f"Unknown microphone backend: {backend}")
        self.thread = threading.Thread(target=self._run, name="voice", daemon=True)
        logging.getLogger(__name__).info(
            "Microphone backend=%s: %s; ASR rate=%s; ASR=%s",
            backend, info["name"], self.rate, cfg.languages,
        )

    def _audio_failed(self, exc):
        self.audio_error = str(exc)
        self.discontinuity.set()

    def _enqueue(self, at: float, block: bytes):
        if self.speaker.input_blocked():
            self.discontinuity.set()
            return
        try:
            self.queue.put_nowait((at, block))
        except queue.Full:
            self.dropped += 1
            self.discontinuity.set()

    def _capture_i2s(self, at: float, block: bytes):
        # Already 16 kHz mono int16 after the hardware adapter's S32_LE conversion.
        self._enqueue(at, block)

    def _capture(self, data, frames, timing, status):
        # Desktop PortAudio callback: copy only, no ASR work here.
        if status:
            self.discontinuity.set()
        block = bytes(data)
        if self.cfg.channels > 1:
            block = np.frombuffer(block, dtype=np.int16).reshape(-1, self.cfg.channels)[:,
                self.cfg.channel_index].copy().tobytes()
        self._enqueue(time.monotonic(), block)

    def start(self):
        self.thread.start()
        self.stream.start()
        return self

    def _run(self):
        started = last_voice = None
        spoken_samples = 0
        pieces = {lang: [] for lang in self.recognizers}
        preroll = deque(maxlen=max(1, 200 // self.cfg.block_ms))
        log = logging.getLogger(__name__)

        def reset():
            nonlocal started, last_voice, spoken_samples, pieces
            for rec in self.recognizers.values():
                rec.Reset()
            pieces = {lang: [] for lang in self.recognizers}
            started = last_voice = None
            spoken_samples = 0
            preroll.clear()
            self.speaker.user_speaking.clear()

        while not self.done.is_set():
            if self.audio_error:
                log.error("Microphone stream failed: %s", self.audio_error)
                break
            try:
                at, block = self.queue.get(timeout=0.10)
            except queue.Empty:
                if self.speaker.input_blocked():
                    reset()
                continue
            if self.speaker.input_blocked() or self.discontinuity.is_set() or time.monotonic() - at > 1.0:
                self.discontinuity.clear()
                reset()
                while True:
                    try:
                        self.queue.get_nowait()
                    except queue.Empty:
                        break
                continue
            audio = np.frombuffer(block, np.int16).astype(np.float32)
            rms = float(np.sqrt(np.mean(audio * audio))) if len(audio) else 0.0
            voiced = rms >= self.cfg.rms_threshold
            if started is None:
                preroll.append(block)
                if not voiced:
                    continue
                started = at - len(preroll) * self.cfg.block_ms / 1000
                self.speaker.user_speaking.set()
                input_blocks = list(preroll)
                preroll.clear()
            else:
                input_blocks = [block]
            if voiced:
                last_voice = at
                spoken_samples += len(audio)
            for chunk in input_blocks:
                for lang, rec in self.recognizers.items():
                    if rec.AcceptWaveform(chunk):
                        pieces[lang].append(json.loads(rec.Result()))
            end = last_voice is not None and at - last_voice >= self.cfg.silence_s
            too_long = at - started >= self.cfg.max_utterance_s
            if not end and not too_long:
                continue
            candidates = []
            for lang, rec in self.recognizers.items():
                parts = pieces[lang] + [json.loads(rec.FinalResult())]
                text = " ".join(part.get("text", "") for part in parts).strip()
                words = [w for part in parts for w in part.get("result", [])]
                confidence = (sum(float(w.get("conf", 0)) for w in words) / len(words)) if words else 0.0
                if self.log_text:
                    log.info("ASR %s %.2f: %s", lang, confidence, text)
                candidates.append((text, lang, confidence))
            command = None if too_long or spoken_samples / self.rate < self.cfg.min_speech_s else select_candidate(
                candidates, self.cfg.confidence, self.cfg.candidate_margin, self.cfg.require_wake_prefix)
            ended = last_voice or at
            reset()
            if command:
                try:
                    self.callback(command, ended)
                except Exception:
                    log.exception("Command callback failed")
        reset()

    def close(self):
        self.done.set()
        close = getattr(self.stream, "close", None)
        stop = getattr(self.stream, "stop", None)
        if stop:
            try:
                stop()
            except Exception:
                pass
        if close:
            try:
                close()
            except Exception:
                pass
        if self.thread.is_alive():
            self.thread.join(timeout=3)
