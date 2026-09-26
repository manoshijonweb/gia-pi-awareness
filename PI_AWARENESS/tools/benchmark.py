#!/usr/bin/env python3
"""Benchmark actual local vision/sensor feature time, without ASR or spoken audio.
Never label these numbers end-to-end voice latency. Use logs plus acoustic timing
for the user's under-three-second requirement. No images/text results are saved.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
from pathlib import Path
import platform
import resource
import subprocess
import importlib.metadata
import statistics
import sys
import time
import cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from settings import CONFIG
from awareness.detector_profiles import PROFILES, apply_profile
from awareness.camera import ImageCamera, capture_once
from awareness.common import install_offline_guard,atomic_json
from awareness.colors import identify_color
from awareness.distance import describe_distance
from awareness.tof import DistanceSensor


def percentile(values,p):
    values=sorted(values)
    index=(len(values)-1)*p
    lower=int(index);upper=min(len(values)-1,lower+1)
    return values[lower]+(values[upper]-values[lower])*(index-lower)


def thermal_snapshot():
    result = {}
    for command in ("measure_temp", "get_throttled", "measure_clock arm"):
        try:
            result[command] = subprocess.check_output(["vcgencmd", *command.split()], text=True, timeout=2).strip()
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(("MemTotal:", "MemAvailable:", "SwapTotal:", "SwapFree:")):
                result[line.split(":")[0] + "_mib"] = int(line.split()[1]) / 1024
    except OSError:
        pass
    return result


def summary(values):
    return {"iterations": len(values), "median_s": statistics.median(values),
            "p95_s": percentile(values, .95), "maximum_s": max(values)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',nargs='+',choices=('color','scan','read','distance'),default=['color','scan','read','distance'])
    p.add_argument('--iterations',type=int,default=10)
    p.add_argument('--image',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'state/benchmark.json')
    p.add_argument('--detector', choices=tuple(PROFILES), help='Override model for this benchmark; no downloads')
    p.add_argument('--threads', type=int, choices=(1,2,3,4), help='Detector ONNX threads only, not OCR')
    args=p.parse_args()
    if args.detector: apply_profile(CONFIG.vision, args.detector)
    if args.threads: CONFIG.vision.object_threads=args.threads
    if args.iterations<1: p.error('--iterations must be positive')
    install_offline_guard();cv2.setNumThreads(1)
    camera=ImageCamera(args.image) if args.image else None
    sensor=DistanceSensor(CONFIG.distance)
    report={'schema':2,'platform':platform.platform(),'machine':platform.machine(),
            'python':platform.python_version(),'end_to_end_voice_latency':False,
            'input':'explicit image' if args.image else 'live camera on-demand per command','features':{},
            'asr_running':False,'tts_running':False,'thermal_start':thermal_snapshot()}
    try:
        if camera is not None and any(f!='distance' for f in args.features): camera.start()
        detector=reader=None
        if 'scan' in args.features:
            from awareness.objects import ObjectDetector
            detector=ObjectDetector(CONFIG.vision);detector.warmup()
            from tools.download_models import sha256
            report['detector']={'name':CONFIG.vision.detector,'format':detector.output_format,
                'input_size':[detector.h,detector.w],'precision':'FP32','threads':detector.threads,
                'model_bytes':CONFIG.vision.object_model.stat().st_size,
                'model_sha256':sha256(CONFIG.vision.object_model),
                'onnxruntime':importlib.metadata.version('onnxruntime')}
        if 'read' in args.features:
            from awareness.ocr import TextReader
            reader=TextReader(CONFIG.vision);reader.warmup()
        if 'distance' in args.features:
            sensor.start()  # on-demand profile: no background polling
        for name in args.features:
            elapsed=[];valid=[];stages={}
            for _ in range(args.iterations):
                t=time.perf_counter()
                if name=='distance':
                    reading=sensor.get();valid.append(reading.valid)
                    describe_distance(reading,CONFIG.distance.default_step_mm)
                else:
                    image=camera.snapshot() if camera is not None else capture_once(CONFIG.camera)
                    if name=='color': identify_color(image,CONFIG.vision)
                    elif name=='scan':
                        detector.detect(image)
                        for stage,value in detector.last_timings.items():
                            stages.setdefault(stage,[]).append(value)
                    elif name=='read':
                        result=reader.read(image);valid.append(not result.incomplete)
                elapsed.append(time.perf_counter()-t)
            stats=summary(elapsed)
            if stages: stats['detector_stages']={name:summary(values) for name,values in stages.items()}
            if valid: stats['valid_or_complete_results']=sum(valid)
            report['features'][name]=stats
            print(name,stats,flush=True)
        report['thermal_end']=thermal_snapshot()
        report['process_peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        atomic_json(args.output,report)
        print(f'Saved {args.output}; live-camera runs include per-command camera open/warmup/close, but exclude ASR, TTS and process startup.')
    finally:
        sensor.close()
        if camera is not None:
            camera.close()
if __name__=='__main__': main()
