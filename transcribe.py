#!/usr/bin/env python3
"""Speech to text (Vosk, offline -- no internet, no API key).

    from inmp441 import Microphone
    from transcribe import SpeechToText

    ears = SpeechToText()
    words = ears.listen(Microphone(), seconds=5)
    print(words)

Run this file directly to record and transcribe once.

Setup:
    pip install vosk --break-system-packages
    wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
    unzip vosk-model-small-en-us-0.15.zip

The small model is fast but rough -- it fills gaps with plausible-sounding
English. vosk-model-en-us-0.22 is much more accurate at about 1.8GB.
"""

import json
import os

from vosk import Model, KaldiRecognizer, SetLogLevel

MODEL_PATH = os.path.expanduser("~/vosk-model-small-en-us-0.15")
RATE = 16000  # what the model expects


class SpeechToText:
    """Turns recorded audio into words."""

    def __init__(self, model_path=MODEL_PATH):
        SetLogLevel(-1)  # stop Vosk printing its startup noise

        if not os.path.isdir(model_path):
            raise RuntimeError(
                f"No Vosk model at {model_path}. Download it with:\n"
                "  wget https://alphacephei.com/vosk/models/"
                "vosk-model-small-en-us-0.15.zip\n"
                "  unzip vosk-model-small-en-us-0.15.zip"
            )
        self.model = Model(model_path)

    def transcribe(self, pcm):
        """Turn 16kHz 16-bit PCM bytes into text."""
        recogniser = KaldiRecognizer(self.model, RATE)

        words = []
        for start in range(0, len(pcm), 4000):
            chunk = pcm[start:start + 4000]
            if recogniser.AcceptWaveform(chunk):
                words.append(json.loads(recogniser.Result())["text"])

        words.append(json.loads(recogniser.FinalResult())["text"])
        return " ".join(part for part in words if part).strip()

    def listen(self, microphone, seconds=5):
        """Record from a Microphone and return what was said."""
        return self.transcribe(microphone.speech_pcm(seconds, rate=RATE))


if __name__ == "__main__":
    from inmp441 import Microphone
    from max98357a import Speaker

    ears = SpeechToText()
    mic = Microphone()

    Speaker().beep()  # the cue: start talking now
    print("Listening for 5 seconds...")

    said = ears.listen(mic, seconds=5)
    print(f"\n  heard: {said or '(nothing recognisable)'}\n")
