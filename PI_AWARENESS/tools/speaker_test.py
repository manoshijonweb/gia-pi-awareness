#!/usr/bin/env python3
"""Speak an English/Hindi test through the configured local speaker."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from settings import CONFIG
from awareness.speech import Speaker
from awareness.common import install_offline_guard
if __name__=='__main__':
    install_offline_guard()
    speaker=Speaker(CONFIG.voice)
    try:
        speaker.say('Speaker test. This device works offline.', 'en')
        speaker.say('यह आवाज़ की जांच है। यह उपकरण बिना इंटरनेट के काम करता है।', 'hi')
        speaker.wait_idle()
    finally:
        speaker.close()
