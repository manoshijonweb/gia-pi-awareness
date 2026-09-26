#!/usr/bin/env python3
"""Live microphone RMS diagnostic. No recordings or transcripts are saved."""
from pathlib import Path
import argparse
import queue
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from settings import CONFIG
from awareness.common import install_offline_guard


def show(audio, threshold, extra=""):
    a=np.frombuffer(audio,np.int16).astype(np.float32)
    rms=float(np.sqrt(np.mean(a*a))) if a.size else 0.0
    peak=float(np.max(np.abs(a))) if a.size else 0.0
    print(f'RMS {rms:8.1f}  peak {peak:7.0f}  {"VOICE" if rms>=threshold else "quiet"}  {extra}',flush=True)
    return rms


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seconds',type=float,default=15)
    a=p.parse_args();install_offline_guard();levels=[]
    print(f'backend={CONFIG.voice.backend}; threshold={CONFIG.voice.rms_threshold:g}')
    print('Remain quiet first, then speak normally. Stop pi-awareness.service first if it is using the I2S mic.')
    end=time.monotonic()+a.seconds
    if CONFIG.voice.backend=='inmp441_i2s':
        from hardware.inmp441 import INMP441Stream
        q=queue.Queue(maxsize=16);errors=[]
        stream=INMP441Stream(block_ms=250,output_rate=16000,channel_index=CONFIG.voice.channel_index,
                             gain=CONFIG.voice.i2s_gain,
                             on_block=lambda at,pcm:q.put_nowait(pcm) if not q.full() else None,
                             on_error=lambda exc:errors.append(str(exc)))
        stream.start()
        try:
            while time.monotonic()<end:
                if errors: raise RuntimeError(errors[-1])
                try:pcm=q.get(timeout=.5)
                except queue.Empty:continue
                levels.append(show(pcm,CONFIG.voice.rms_threshold,f'gain={CONFIG.voice.i2s_gain:g}'))
        finally:stream.close()
    else:
        import sounddevice as sd
        from awareness.voice import choose_input
        rate,info=choose_input(CONFIG.voice);print(f'{info["name"]}: {rate} Hz')
        with sd.RawInputStream(device=CONFIG.voice.microphone,channels=CONFIG.voice.channels,
                               samplerate=rate,dtype='int16') as stream:
            while time.monotonic()<end:
                data,overflow=stream.read(int(rate*.25))
                audio=np.frombuffer(data,np.int16).reshape(-1,CONFIG.voice.channels)[:,CONFIG.voice.channel_index].copy().tobytes()
                levels.append(show(audio,CONFIG.voice.rms_threshold,'OVERFLOW' if overflow else ''))
    if levels:print(f'Observed RMS p10/p50/p90: {np.percentile(levels,[10,50,90]).round(1)}')
    print('Tune voice.rms_threshold and, for INMP441, voice.i2s_gain in settings.py if needed.')
if __name__=='__main__':main()
