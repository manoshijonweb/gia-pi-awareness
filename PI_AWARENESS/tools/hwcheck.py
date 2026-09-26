#!/usr/bin/env python3
"""Quick check of the exact assembled Pi 5 hardware profile.

Stop the running service first so the I2S capture PCM is not already open:
  sudo systemctl stop pi-awareness
  .venv/bin/python tools/hwcheck.py
  sudo systemctl start pi-awareness
"""
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from settings import CONFIG


def run(cmd):
    p=subprocess.run(cmd,text=True,capture_output=True)
    return p.returncode,(p.stdout or '')+(p.stderr or '')

def main():
    failures=0
    print('PI_AWARENESS hardware check')
    print('===========================')
    model=Path('/proc/device-tree/model')
    text=model.read_text(errors='ignore').strip('\0') if model.exists() else 'unknown'
    print('Pi:',text)

    from hardware.i2s_common import find_i2s_device,overlay_is_configured
    print('I2S overlay configured:',overlay_is_configured())
    try:
        dev=find_i2s_device();print('I2S PCM:',dev)
    except Exception as exc:
        failures+=1;print('FAIL I2S:',exc);dev=None

    if dev:
        try:
            from hardware.inmp441 import Microphone
            pcm=Microphone(dev).speech_pcm(1.0,gain=CONFIG.voice.i2s_gain)
            a=np.frombuffer(pcm,dtype='<i2').astype(np.float32)
            rms=float(np.sqrt(np.mean(a*a))) if a.size else 0.0
            print(f'INMP441: RMS={rms:.1f} after configured gain={CONFIG.voice.i2s_gain:g}')
            if rms==0:failures+=1;print('FAIL INMP441 digital silence')
        except Exception as exc:
            failures+=1;print('FAIL INMP441:',exc)
        try:
            from hardware.max98357a import Speaker
            print('MAX98357A: playing 0.25 second test beep')
            Speaker(dev).beep()
        except Exception as exc:
            failures+=1;print('FAIL MAX98357A:',exc)

    try:
        from awareness.camera import capture_once
        frame=capture_once(CONFIG.camera)
        print('Camera:',frame.shape)
    except Exception as exc:
        failures+=1;print('FAIL camera:',exc)

    try:
        from awareness.tof import VL53L0X
        tof=VL53L0X(CONFIG.distance)
        try:r=tof.sample()
        finally:tof.close()
        print('VL53L0X:',f'{r.mm:g} mm' if r.valid else r.reason)
    except Exception as exc:
        failures+=1;print('FAIL VL53L0X:',exc)

    print(f'\n{failures} failure(s)')
    return 1 if failures else 0

if __name__=='__main__':raise SystemExit(main())
