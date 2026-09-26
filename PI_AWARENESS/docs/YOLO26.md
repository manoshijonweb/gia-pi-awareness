# YOLO26 detector profile

Default: YOLO26m, FP32 ONNX, 640×640, batch 1, CPU ONNX Runtime.

The ready-made default file is the pinned ONNX Community conversion in `tools/model_manifest.py`.
It uses named `logits` and `pred_boxes` outputs and direct RGB resize. The adapter validates the graph contract instead
of guessing from a `.onnx` filename.

Smaller explicit profiles remain available:

```bash
.venv/bin/python tools/download_models.py --detector yolo26s --only-detector
.venv/bin/python server.py --detector yolo26s --once scan
```

or `yolo26n`.

The service default remains YOLO26m. No runtime fallback silently changes the model.

A native Ultralytics export helper is retained in `tools/export_yolo26.py` for workstation-only experiments.
Do not install PyTorch/Ultralytics on the Pi merely to run the service.
