# Offline model directory

Model weights are not bundled in the source ZIP. `bash install.sh` downloads them once during provisioning.
Normal runtime does not download or replace models.

Expected default assets:

```text
yolo26m_640.onnx
ocr_det.onnx
ocr_rec.onnx
ocr_cls.onnx
vosk-model-small-en-in-0.4/...
vosk-model-small-hi-0.22/...
installed_manifest.json
```

Verify local assets without network access:

```bash
.venv/bin/python tools/download_models.py --verify
```

Smaller detector profiles can be provisioned explicitly with `--detector yolo26s --only-detector` or `yolo26n`.
No runtime model fallback is performed.
