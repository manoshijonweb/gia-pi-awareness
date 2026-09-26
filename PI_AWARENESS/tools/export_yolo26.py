#!/usr/bin/env python3
"""Optional WORKSTATION-only official Ultralytics -> native ONNX export.
Not invoked by install.sh or the Pi runtime. The default ready-made community
model needs no export. Install ultralytics, onnx, onnxruntime on the workstation.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]


def validate_contract(shape, outputs, size):
    if list(shape) != [1,3,size,size]:
        raise ValueError(f'Expected static FP32 NCHW [1,3,{size},{size}], got {shape}')
    if len(outputs)!=1 or len(outputs[0].shape)!=3 or outputs[0].shape[0]!=1 or outputs[0].shape[-1]!=6:
        raise ValueError('Expected native processed [1,N,6] COCO detection output. Export was not e2e; do not rename an incompatible file.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',default='yolo26m.pt',help='Official YOLO26 COCO detect checkpoint or a trusted local equivalent')
    parser.add_argument('--imgsz',type=int,default=640)
    parser.add_argument('--output',type=Path,default=Path('yolo26m_native_640.onnx'))
    args=parser.parse_args()
    if args.imgsz<320 or args.imgsz%32:
        parser.error('--imgsz must be >=320 and divisible by 32')
    try:
        from ultralytics import YOLO
        import numpy as np
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit('Run this on a separate workstation: pip install ultralytics onnx onnxruntime. Do NOT add these export tools to the Pi installer.') from exc
    sys.path.insert(0,str(ROOT))
    from awareness.objects import COCO
    model=YOLO(args.weights)
    if model.task!='detect' or tuple(model.names[i] for i in range(len(model.names)))!=COCO:
        raise SystemExit('This application needs the standard 80 COCO detection classes in standard order.')
    result=Path(model.export(format='onnx',imgsz=args.imgsz,batch=1,dynamic=False,
                            half=False,nms=False,simplify=False,opset=17,device='cpu'))
    graph=onnx.load(str(result));onnx.checker.check_model(graph)
    sess=ort.InferenceSession(str(result),providers=['CPUExecutionProvider'])
    if len(sess.get_inputs())!=1 or sess.get_inputs()[0].type!='tensor(float)':
        raise SystemExit('Export does not have a single FP32 input.')
    outputs=sess.run(None,{sess.get_inputs()[0].name:np.zeros((1,3,args.imgsz,args.imgsz),np.float32)})
    validate_contract(sess.get_inputs()[0].shape,outputs,args.imgsz)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if result.resolve()!=args.output.resolve():shutil.copy2(result,args.output)
    digest=hashlib.sha256(args.output.read_bytes()).hexdigest()
    metadata={'weights':args.weights,'onnx_sha256':digest,'ultralytics':importlib.metadata.version('ultralytics'),
              'onnxruntime':ort.__version__,'input':[1,3,args.imgsz,args.imgsz],
              'output':list(outputs[0].shape),'format':'ultralytics_e2e','precision':'FP32',
              'note':'Workstation smoke test, not a Pi benchmark or an accuracy validation.'}
    args.output.with_suffix('.export.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Export validated: {args.output}\nSHA256: {digest}')
    print('Copy the ONNX file to your Pi models directory and set VisionSettings:')
    print('  detector = "yolo26m"  # use the actual YOLO26 scale you exported')
    print('  object_format = "ultralytics_e2e"')
    print(f'  object_model = ROOT / "models/{args.output.name}"\n  object_input_size = {args.imgsz}')
    print('Do not use --detector profiles with this custom export; those select the community files.')
    print('Provision other assets with tools/download_models.py --skip-detector; run deep doctor and real-scene tests.')

if __name__=='__main__':main()
