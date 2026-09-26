"""CPU ONNX object detection with explicitly matched image/output contracts.
YOLO26 community: RGB 0..1, direct resize, logits + normalized cxcywh outputs.
YOLO26 Ultralytics: RGB 0..1, centered letterbox, explicit e2e xyxy OR raw xywh.
YOLOX: BGR 0..255, top-left letterbox, raw head decoding.
The community export is NOT the same output contract as an Ultralytics export.
No Ultralytics, transformers, torch, or additional inference runtime is imported.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import cv2
import numpy as np
from .common import FeatureError
from .ort import session, infer, fixed_dimension

COCO = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)
COCO_HI = (
    "व्यक्ति", "साइकिल", "कार", "मोटरसाइकिल", "हवाई जहाज़", "बस", "रेलगाड़ी", "ट्रक", "नाव",
    "ट्रैफिक लाइट", "फायर हाइड्रेंट", "रुकने का संकेत", "पार्किंग मीटर", "बेंच", "पक्षी", "बिल्ली", "कुत्ता",
    "घोड़ा", "भेड़", "गाय", "हाथी", "भालू", "ज़ेब्रा", "जिराफ़", "बैग", "छाता",
    "हैंडबैग", "टाई", "सूटकेस", "फ्रिसबी", "स्की", "स्नोबोर्ड", "गेंद", "पतंग",
    "बेसबॉल का बल्ला", "बेसबॉल का दस्ताना", "स्केटबोर्ड", "सर्फबोर्ड", "टेनिस रैकेट", "बोतल",
    "वाइन का गिलास", "कप", "कांटा", "चाकू", "चम्मच", "कटोरा", "केला", "सेब", "सैंडविच",
    "संतरा", "ब्रोकली", "गाजर", "हॉट डॉग", "पिज़्ज़ा", "डोनट", "केक", "कुर्सी", "सोफ़ा",
    "गमले का पौधा", "बिस्तर", "खाने की मेज़", "शौचालय", "टीवी", "लैपटॉप", "माउस", "रिमोट",
    "कीबोर्ड", "मोबाइल फ़ोन", "माइक्रोवेव", "ओवन", "टोस्टर", "सिंक", "फ्रिज", "किताब",
    "घड़ी", "फूलदान", "कैंची", "टेडी बियर", "हेयर ड्रायर", "टूथब्रश",
)

@dataclass(frozen=True)
class Detection:
    class_id: int
    score: float
    box: tuple[float, float, float, float]


def letterbox(image: np.ndarray, height: int, width: int, centered: bool):
    h, w = image.shape[:2]
    scale = min(height / h, width / w)
    # Ultralytics rounds resized dimensions; legacy YOLOX truncates them.
    rounding = round if centered else int
    rh, rw = max(1, rounding(h * scale)), max(1, rounding(w * scale))
    left, top = ((width - rw) // 2, (height - rh) // 2) if centered else (0, 0)
    padded = np.full((height, width, 3), 114, dtype=np.uint8)
    padded[top:top + rh, left:left + rw] = cv2.resize(image, (rw, rh))
    return padded, scale, left, top


def nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou: float) -> list[int]:
    order = np.argsort(scores)[::-1][:1000]
    result = []
    while len(order):
        chosen = int(order[0])
        result.append(chosen)
        others = order[1:]
        if not len(others):
            break
        a = boxes[chosen]
        b = boxes[others]
        lt = np.maximum(a[:2], b[:, :2])
        rb = np.minimum(a[2:], b[:, 2:])
        inter = np.prod(np.maximum(0.0, rb - lt), axis=1)
        area_a = np.prod(np.maximum(0.0, a[2:] - a[:2]))
        area_b = np.prod(np.maximum(0.0, b[:, 2:] - b[:, :2]), axis=1)
        overlap = inter / np.maximum(area_a + area_b - inter, 1e-8)
        order = others[(overlap <= iou) | (classes[others] != classes[chosen])]
    return result


def decode_yolox(raw: np.ndarray, height: int, width: int):
    prediction = np.asarray(raw, dtype=np.float32)
    if prediction.ndim != 3 or prediction.shape[0] != 1 or prediction.shape[2] != 85:
        raise FeatureError(f"Wrong YOLOX output shape: {prediction.shape}")
    p = prediction[0].copy()
    grids, strides = [], []
    for stride in (8, 16, 32):
        xx, yy = np.meshgrid(np.arange(width // stride), np.arange(height // stride))
        grids.append(np.stack((xx, yy), axis=-1).reshape(-1, 2))
        strides.append(np.full((xx.size, 1), stride))
    grid, stride = np.concatenate(grids), np.concatenate(strides)
    if len(p) != len(grid):
        raise FeatureError("YOLOX model/output grid mismatch; use the official raw-head export")
    xy = (p[:, :2] + grid) * stride
    wh = np.exp(np.clip(p[:, 2:4], -12, 12)) * stride
    boxes = np.concatenate((xy - wh / 2, xy + wh / 2), axis=1)
    score_matrix = p[:, 4:5] * p[:, 5:]
    ids = score_matrix.argmax(axis=1)
    return boxes, score_matrix[np.arange(len(ids)), ids], ids


def decode_yolov8(raw: np.ndarray):
    p = np.asarray(raw)
    if p.ndim != 3 or p.shape[0] != 1:
        raise FeatureError(f"Wrong YOLOv8 output shape: {p.shape}")
    p = p[0]
    if p.shape[0] == 84:
        p = p.T
    if p.shape[1] != 84:
        raise FeatureError("YOLOv8 requires the COCO detect export with nms=False, not pose/seg/end-to-end")
    boxes = np.concatenate((p[:, :2] - p[:, 2:4] / 2, p[:, :2] + p[:, 2:4] / 2), axis=1)
    ids = p[:, 4:].argmax(axis=1)
    return boxes, p[np.arange(len(ids)), ids + 4], ids

def decode_yolo26_e2e(raw: np.ndarray):
    """Ultralytics processed COCO rows [x1,y1,x2,y2,confidence,class_id].

    Both NMS-free and embedded-NMS exports are already postprocessed. Do not
    run another NMS here. Malformed rows are rejected before casting class IDs.
    """
    p = np.asarray(raw, dtype=np.float32)
    if p.ndim != 3 or p.shape[0] != 1 or p.shape[2] != 6:
        raise FeatureError(f"YOLO26 e2e needs [1,N,6], received {p.shape}; check object_format")
    p = p[0]
    finite = np.isfinite(p).all(axis=1)
    q = p[finite]
    valid = ((q[:, 4] >= 0) & (q[:, 4] <= 1) & (q[:, 5] >= 0)
             & (q[:, 5] < 80) & (q[:, 5] == np.floor(q[:, 5])))
    q = q[valid]
    return q[:, :4].copy(), q[:, 4].copy(), q[:, 5].astype(np.int64)


def decode_yolo26_community(logits: np.ndarray, pred_boxes: np.ndarray, height: int, width: int):
    """ONNX Community YOLO26: sigmoid class logits, normalized center-format boxes.

    This export's publisher example uses 80 independent sigmoid scores, NOT
    softmax, and does NOT apply NMS. Infinite logits have defined sigmoid limits;
    rows containing NaN logits or nonfinite box coordinates are discarded.
    """
    log = np.asarray(logits, dtype=np.float32)
    box = np.asarray(pred_boxes, dtype=np.float32)
    if (log.ndim != 3 or log.shape[0] != 1 or log.shape[2] != 80
            or box.ndim != 3 or box.shape[0] != 1 or box.shape[2] != 4
            or log.shape[1] != box.shape[1]):
        raise FeatureError(f"Community YOLO26 needs logits [1,N,80] and pred_boxes [1,N,4], got {log.shape}, {box.shape}")
    good = ~np.isnan(log[0]).any(axis=1) & np.isfinite(box[0]).all(axis=1)
    log, box = log[0][good], box[0][good]
    ids = log.argmax(axis=1)
    values = log[np.arange(len(ids)), ids]
    scores = 1.0 / (1.0 + np.exp(-np.clip(values, -80, 80)))
    boxes = np.concatenate((box[:, :2] - box[:, 2:] / 2, box[:, :2] + box[:, 2:] / 2), axis=1)
    boxes *= np.array([width, height, width, height], dtype=np.float32)
    return boxes, scores, ids


def detector_format(cfg) -> str:
    if cfg.detector == "yolox_nano":
        return "yolox_raw"
    if cfg.detector == "yolov8n":
        return "ultralytics_raw"
    if cfg.detector not in ("yolo26m", "yolo26s", "yolo26n", "yolo26l", "yolo26x"):
        raise FeatureError("Unknown detector in settings.py")
    fmt = getattr(cfg, "object_format", "community_yolos")
    if fmt not in ("community_yolos", "ultralytics_e2e", "ultralytics_raw"):
        raise FeatureError(f"Unknown YOLO26 output format: {fmt}")
    return fmt


class ObjectDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.output_format = detector_format(cfg)
        self.threads = getattr(cfg, "object_threads", cfg.ort_threads)
        self.session = session(cfg.object_model, self.threads)
        shape = self.session.get_inputs()[0].shape
        if len(shape) != 4:
            raise FeatureError("Object detector expects NCHW input")
        if (isinstance(shape[0], int) and shape[0] != 1) or (isinstance(shape[1], int) and shape[1] != 3):
            raise FeatureError("Object detector needs batch 1 and 3 RGB/BGR channels")
        self.h = fixed_dimension(shape[2], cfg.object_input_size)
        self.w = fixed_dimension(shape[3], cfg.object_input_size)
        if (self.h, self.w) != (cfg.object_input_size, cfg.object_input_size):
            raise FeatureError(f"Configured input {cfg.object_input_size} does not match graph {self.h}x{self.w}")
        self.last_timings = {}
        # Explicit names distinguish the community contract from a native export.
        if self.output_format == "community_yolos":
            outputs = {x.name: x for x in self.session.get_outputs()}
            if set(outputs) != {"logits", "pred_boxes"}:
                raise FeatureError("Community YOLO26 expects named logits and pred_boxes outputs. For an official export select ultralytics_e2e or ultralytics_raw.")
            for name, channels in (("logits", 80), ("pred_boxes", 4)):
                out_shape = outputs[name].shape
                if (len(out_shape) != 3 or (isinstance(out_shape[-1], int) and out_shape[-1] != channels)):
                    raise FeatureError(f"Unexpected {name} dimensions: {out_shape}")
        elif len(self.session.get_outputs()) != 1:
            raise FeatureError("This detector adapter expects a single detection output, not segmentation/pose")

    def detect(self, image: np.ndarray) -> list[Detection]:
        start = time.perf_counter()
        if (not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3
                or image.shape[0] < 1 or image.shape[1] < 1 or image.dtype != np.uint8):
            raise FeatureError("Object detector requires a nonempty uint8 BGR camera image")
        fmt = detector_format(self.cfg)
        if fmt == "community_yolos":
            # Match the publisher's do_pad=false and size={height:640,width:640}.
            # OpenCV bilinear is not promised bit-identical to browser/Pillow resize.
            img = cv2.resize(image, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            scale, left, top = 1.0, 0, 0
        else:
            img, scale, left, top = letterbox(image, self.h, self.w, fmt != "yolox_raw")
        if fmt != "yolox_raw":
            img = img[:, :, ::-1].astype(np.float32) / 255.0
        tensor = np.ascontiguousarray(img.transpose(2, 0, 1)[None], dtype=np.float32)
        prepared = time.perf_counter()
        if fmt == "community_yolos":
            outputs = self.session.run(["logits", "pred_boxes"], {self.session.get_inputs()[0].name: tensor})
        else:
            outputs = infer(self.session, tensor)
        inferred = time.perf_counter()
        if fmt == "community_yolos":
            boxes, scores, ids = decode_yolo26_community(*outputs, image.shape[0], image.shape[1])
        elif fmt == "ultralytics_e2e":
            boxes, scores, ids = decode_yolo26_e2e(outputs)
        elif fmt == "ultralytics_raw":
            boxes, scores, ids = decode_yolov8(outputs)
        else:
            boxes, scores, ids = decode_yolox(outputs, self.h, self.w)
        good = (np.isfinite(scores) & np.isfinite(boxes).all(axis=1)
                & (scores >= self.cfg.object_confidence) & (scores <= 1.0) & (ids >= 0) & (ids < len(COCO)))
        boxes, scores, ids = boxes[good], scores[good], ids[good]
        if fmt != "community_yolos":
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / scale
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / scale
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, image.shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, image.shape[0])
        good = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, scores, ids = boxes[good], scores[good], ids[good]
        keep = (np.argsort(scores)[::-1].tolist() if fmt in ("community_yolos", "ultralytics_e2e")
                else nms(boxes, scores, ids, self.cfg.object_iou))
        result = [Detection(int(ids[i]), float(scores[i]), tuple(float(x) for x in boxes[i])) for i in keep]
        ended = time.perf_counter()
        self.last_timings = {"preprocess_s": prepared-start, "inference_s": inferred-prepared,
                             "postprocess_s": ended-inferred, "total_s": ended-start}
        return result

    def warmup(self):
        self.detect(np.zeros((self.h, self.w, 3), np.uint8))


def _plural_en(label: str, count: int) -> str:
    if count == 1:
        return label
    irregular = {
        "person": "people", "mouse": "mice", "knife": "knives",
        "toothbrush": "toothbrushes", "bus": "buses", "sandwich": "sandwiches",
        "wine glass": "wine glasses", "sports ball": "sports balls",
        "traffic light": "traffic lights", "fire hydrant": "fire hydrants",
        "stop sign": "stop signs", "parking meter": "parking meters",
        "dining table": "dining tables", "cell phone": "cell phones",
        "potted plant": "potted plants", "teddy bear": "teddy bears",
        "hair drier": "hair driers", "baseball bat": "baseball bats",
        "baseball glove": "baseball gloves", "tennis racket": "tennis rackets",
    }
    if label in irregular:
        return irregular[label]
    if label.endswith(("s", "x", "ch", "sh")):
        return label + "es"
    if label.endswith("y") and len(label) > 1 and label[-2].lower() not in "aeiou":
        return label[:-1] + "ies"
    return label + "s"


def describe_objects(detections: list[Detection], language: str, limit: int = 8) -> str:
    """Speak class counts, e.g. 'I see 2 people, 1 chair, and 1 bottle.'"""
    counts: dict[int, int] = {}
    best_score: dict[int, float] = {}
    for item in detections:
        if 0 <= item.class_id < len(COCO):
            counts[item.class_id] = counts.get(item.class_id, 0) + 1
            best_score[item.class_id] = max(best_score.get(item.class_id, 0.0), item.score)
    if not counts:
        return ("पहचानी हुई वस्तु नहीं मिली। इसका मतलब रास्ता खाली नहीं है।" if language == "hi"
                else "No familiar objects detected. This does not mean the area is clear.")

    ordered = sorted(counts, key=lambda cid: (-best_score[cid], cid))
    shown = ordered[:max(1, int(limit))]
    parts = []
    if language == "hi":
        for cid in shown:
            parts.append(f"{counts[cid]} {COCO_HI[cid]}")
        joiner = " और "
        prefix = "मुझे "
        suffix = " दिखाई दे रहे हैं।"
    else:
        for cid in shown:
            count = counts[cid]
            parts.append(f"{count} {_plural_en(COCO[cid], count)}")
        joiner = " and "
        prefix = "I see "
        suffix = "."

    if len(parts) == 1:
        joined = parts[0]
    elif len(parts) == 2:
        joined = joiner.join(parts)
    else:
        joined = ", ".join(parts[:-1]) + "," + joiner + parts[-1]
    if len(ordered) > len(shown):
        joined += (" और अन्य वस्तुएँ" if language == "hi" else " and other objects")
    return prefix + joined + suffix
