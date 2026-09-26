#!/usr/bin/env python3
"""OPTIONAL workstation-only export. Never invoked by Pi installer/runtime.
Run in a separate workstation venv with ultralytics + onnx + torch installed.
Review Ultralytics' applicable model/software licenses before distribution.
"""
import argparse
from pathlib import Path
import shutil

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',default='yolov8n.pt')
    p.add_argument('--output',type=Path,default=Path('yolov8n_320.onnx'))
    a=p.parse_args()
    from ultralytics import YOLO
    result=YOLO(a.weights).export(format='onnx',imgsz=320,batch=1,dynamic=False,
                                 simplify=False,opset=13,half=False,nms=False,device='cpu')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if Path(result).resolve()!=a.output.resolve():
        shutil.copy2(result,a.output)
    print(f'Copy {a.output} to Pi models/yolov8n_320.onnx')
    print('Set vision.detector="yolov8n", object_model=ROOT/"models/yolov8n_320.onnx", object_input_size=320 in settings.py.')
    print('Then run server.py --doctor --deep and benchmark your own scenes; models are not accuracy-equivalent.')
if __name__=='__main__':
    main()
