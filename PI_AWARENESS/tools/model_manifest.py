"""Versioned public assets. YOLO26 HF LFS hashes and OCR RapidAI hashes are pinned.
YOLOX/Vosk hashes are NOT independently pinned; installation records local hashes.
Sources and license notes: docs/SOURCES.md and THIRD_PARTY_NOTICES.md.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    target: str
    sha256: str | None = None
    archive_root: str | None = None
    maximum_bytes: int = 256 * 1024 * 1024

BASE = 'https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx'
SHARED_AND_LEGACY = (
    Asset('YOLOX-Nano official ONNX',
          'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx',
          'yolox_nano.onnx'),
    Asset('PP-OCRv5 mobile detector', BASE+'/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx',
          'ocr_det.onnx', '4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae'),
    Asset('PP-OCRv5 Devanagari and English recognizer', BASE+'/PP-OCRv5/rec/devanagari_PP-OCRv5_rec_mobile.onnx',
          'ocr_rec.onnx', 'd6f0a906580e3fa6b324a318718f1f31f268b6ea8ef985f91c2012a37f52c91e'),
    Asset('PP-LCNet mobile text-line orientation classifier',
          BASE+'/PP-OCRv5/cls/ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx',
          'ocr_cls.onnx', '54379ae5174d026780215fc748a7f31910dee36818e63d49e17dc598ecc82df7'),
    Asset('Vosk small Indian English',
          'https://alphacephei.com/vosk/models/vosk-model-small-en-in-0.4.zip',
          'vosk-model-small-en-in-0.4', archive_root='vosk-model-small-en-in-0.4'),
    Asset('Vosk small Hindi',
          'https://alphacephei.com/vosk/models/vosk-model-small-hi-0.22.zip',
          'vosk-model-small-hi-0.22', archive_root='vosk-model-small-hi-0.22'),
)

# These are ONNX Community / Xenova conversions, NOT official Ultralytics exports.
# FP32 only. They have two named outputs (logits, pred_boxes), not [1,300,6].
# Publisher file hashes, immutable revision URLs, preprocessing, and the author's
# working example are documented in docs/YOLO26.md. No HF/transformers dependency.
DETECTOR_ASSETS = {
    "yolo26m": Asset("YOLO26m FP32 640 (ONNX Community)",
        "https://huggingface.co/onnx-community/yolo26m-ONNX/resolve/a1db4877f0a3ed68554c231cdae958e2280087e3/onnx/model.onnx",
        "yolo26m_640.onnx", "7cd89faaa164887c3c33ee0fe5d63723bf262f75dd90088c4cb3be5a310aabff"),
    "yolo26s": Asset("YOLO26s FP32 640 (ONNX Community)",
        "https://huggingface.co/onnx-community/yolo26s-ONNX/resolve/37669b009f416cb1df28751257d6ec5f8e4b4e20/onnx/model.onnx",
        "yolo26s_640.onnx", "c72bc5ad4e7f7c87666a051d0f01fc02d084688c1ca78e825031827a0590efb1"),
    "yolo26n": Asset("YOLO26n FP32 640 (ONNX Community)",
        "https://huggingface.co/onnx-community/yolo26n-ONNX/resolve/a8dc7e14743e1cea8ccd493bd99b4c2827de1acf/onnx/model.onnx",
        "yolo26n_640.onnx", "cda08d9440217e243e075ee839f40383c59b3f973e493f9f6c7452922a69436e"),
    "yolox_nano": SHARED_AND_LEGACY[0],
}
SHARED_ASSETS = SHARED_AND_LEGACY[1:]

def assets_for(detector: str = "yolo26m", only_detector: bool = False, skip_detector: bool = False):
    if only_detector and skip_detector:
        raise ValueError("Cannot both select and skip the detector")
    if skip_detector:
        return SHARED_ASSETS
    if detector not in DETECTOR_ASSETS:
        raise ValueError("No automatic download for this detector; provide your export and use --skip-detector")
    return (DETECTOR_ASSETS[detector],) + (() if only_detector else SHARED_ASSETS)

ASSETS = assets_for()
