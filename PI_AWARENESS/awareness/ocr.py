"""Lean DB + CTC pipeline for RapidAI-converted PaddleOCR mobile ONNX models.
No PaddlePaddle, PyTorch, RapidOCR package, Shapely or model-download code at runtime.
English + Hindi use one PP-OCRv5 Devanagari recognizer with embedded character dict.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import threading
import time
import cv2
import numpy as np
from .common import FeatureError, Cancelled
from .ort import session, infer, fixed_dimension

@dataclass(frozen=True)
class TextLine:
    text: str
    confidence: float
    box: tuple[tuple[float, float], ...]

@dataclass(frozen=True)
class OCRResult:
    lines: list[TextLine]
    incomplete: bool
    detected_regions: int
    rejected_regions: int


def order_quad(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, np.float32).reshape(4, 2)
    # Sort by x, then resolve left/right y; no duplicate argmin corners on diamonds.
    by_x = p[np.argsort(p[:, 0], kind="stable")]
    left = by_x[:2][np.argsort(by_x[:2, 1])]
    right = by_x[2:][np.argsort(by_x[2:, 1])]
    return np.array([left[0], right[0], right[1], left[1]], np.float32)


def reading_order(boxes: list[np.ndarray]) -> list[np.ndarray]:
    # Row clustering for ordinary upright labels/pages. Not a multi-column layout engine.
    rows: list[list[np.ndarray]] = []
    for box in sorted(boxes, key=lambda b: float(b[:, 1].mean())):
        cy = float(box[:, 1].mean())
        height = max(1.0, float(np.linalg.norm(box[3] - box[0])))
        placed = False
        for row in reversed(rows[-3:]):
            rcy = float(np.mean([b[:, 1].mean() for b in row]))
            rh = float(np.median([np.linalg.norm(b[3] - b[0]) for b in row]))
            if abs(cy - rcy) < max(height, rh) * 0.45:
                row.append(box)
                placed = True
                break
        if not placed:
            rows.append([box])
    return [b for row in rows for b in sorted(row, key=lambda b: float(b[:, 0].min()))]


def db_boxes(probability: np.ndarray, image_shape: tuple[int, int], cfg) -> list[np.ndarray]:
    p = np.asarray(probability, np.float32).squeeze()
    if p.ndim != 2 or not np.isfinite(p).all():
        raise FeatureError(f"Invalid DB text detector output: {p.shape}")
    mask = (p > cfg.ocr_bitmap_threshold).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    # Candidate cap is defensive; this is not exhaustive page layout extraction.
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:1500]:
        rect = cv2.minAreaRect(contour)
        (cx, cy), (width, height), angle = rect
        if min(width, height) < 3:
            continue
        quad = cv2.boxPoints(rect)
        x, y, w, h = cv2.boundingRect(quad.astype(np.int32))
        x0, y0, x1, y1 = max(0, x), max(0, y), min(p.shape[1], x+w), min(p.shape[0], y+h)
        if x1 <= x0 or y1 <= y0:
            continue
        local = np.zeros((y1-y0, x1-x0), np.uint8)
        cv2.fillPoly(local, [np.round(quad - [x0, y0]).astype(np.int32)], 1)
        score = cv2.mean(p[y0:y1, x0:x1], mask=local)[0]
        if score < cfg.ocr_box_threshold:
            continue
        # DB unclip on a rectangle: expanding each side by area*ratio/perimeter.
        # The resulting minimum-area rectangle is the analytic offset rectangle;
        # no polygon-clipping dependency is required for this quadrilateral path.
        expansion = width * height * cfg.ocr_unclip_ratio / (2 * (width + height))
        quad = order_quad(cv2.boxPoints(((cx, cy), (width+2*expansion, height+2*expansion), angle)))
        quad[:, 0] *= image_shape[1] / p.shape[1]
        quad[:, 1] *= image_shape[0] / p.shape[0]
        quad[:, 0] = quad[:, 0].clip(0, image_shape[1]-1)
        quad[:, 1] = quad[:, 1].clip(0, image_shape[0]-1)
        boxes.append(quad)
    return reading_order(boxes)


def crop_quad(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    box = order_quad(box)
    width = max(2, int(max(np.linalg.norm(box[1]-box[0]), np.linalg.norm(box[2]-box[3]))))
    height = max(2, int(max(np.linalg.norm(box[3]-box[0]), np.linalg.norm(box[2]-box[1]))))
    destination = np.array([[0,0], [width-1,0], [width-1,height-1], [0,height-1]], np.float32)
    matrix = cv2.getPerspectiveTransform(box, destination)
    crop = cv2.warpPerspective(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    if height / width >= 1.5:
        crop = np.rot90(crop).copy()
    return crop


def normalize_line(image: np.ndarray, height: int, width: int) -> np.ndarray:
    resized_width = max(1, min(width, math.ceil(height * image.shape[1] / image.shape[0])))
    resized = cv2.resize(image, (resized_width, height)).astype(np.float32)
    out = np.zeros((1, 3, height, width), np.float32)
    out[0, :, :, :resized_width] = (resized.transpose(2,0,1) / 255.0 - 0.5) / 0.5
    return out


def ctc_decode(prediction: np.ndarray, characters: list[str]) -> tuple[str, float]:
    p = np.asarray(prediction)
    if p.ndim != 3 or p.shape[0] != 1 or p.shape[2] != len(characters):
        raise FeatureError(f"OCR output/dictionary mismatch: {p.shape}, {len(characters)} entries")
    if not np.isfinite(p).all():
        raise FeatureError("Non-finite OCR output")
    # Official models already emit probabilities, not logits.
    ids = p[0].argmax(axis=1)
    probs = p[0].max(axis=1)
    selected = ids != 0
    selected[1:] &= ids[1:] != ids[:-1]
    if not selected.any():
        return "", 0.0
    return "".join(characters[i] for i in ids[selected]).strip(), float(probs[selected].mean())

class TextReader:
    def __init__(self, cfg):
        self.cfg = cfg
        self.det = session(cfg.ocr_det, cfg.ort_threads)
        self.rec = session(cfg.ocr_rec, cfg.ort_threads)
        self.cls = session(cfg.ocr_cls, cfg.ort_threads) if cfg.ocr_use_cls else None
        metadata = self.rec.get_modelmeta().custom_metadata_map
        if not metadata.get("character"):
            raise FeatureError("OCR model has no embedded character dictionary; use the supplied model manifest")
        self.characters = ["blank"] + metadata["character"].splitlines() + [" "]
        shape = self.rec.get_inputs()[0].shape
        self.rec_h = fixed_dimension(shape[2], cfg.ocr_rec_height)
        self.fixed_rec_w = shape[3] if isinstance(shape[3], int) and shape[3] > 0 else None
        if self.cls is not None:
            shape = self.cls.get_inputs()[0].shape
            self.cls_h = fixed_dimension(shape[2], 80)
            self.cls_w = fixed_dimension(shape[3], 160)

    def detect(self, image: np.ndarray) -> list[np.ndarray]:
        h, w = image.shape[:2]
        scale = min(1.0, self.cfg.ocr_det_limit / max(h, w))
        shape = self.det.get_inputs()[0].shape
        dh = fixed_dimension(shape[2], max(32, round(h*scale/32)*32))
        dw = fixed_dimension(shape[3], max(32, round(w*scale/32)*32))
        tensor = cv2.resize(image, (dw, dh)).astype(np.float32) / 255.0
        tensor = (tensor - np.array([0.485,0.456,0.406], np.float32)) / np.array([0.229,0.224,0.225], np.float32)
        output = infer(self.det, np.ascontiguousarray(tensor.transpose(2,0,1)[None]))
        return db_boxes(output, (h,w), self.cfg)

    def recognize(self, crop: np.ndarray) -> tuple[str, float]:
        if self.cls is not None:
            orientation = np.asarray(infer(self.cls, normalize_line(crop, self.cls_h, self.cls_w))).reshape(-1)
            if len(orientation) != 2:
                raise FeatureError("Expected 0/180-degree two-class text orientation model")
            if orientation[1] > 0.90 and orientation[1] > orientation[0]:
                crop = cv2.rotate(crop, cv2.ROTATE_180)
        required = max(self.cfg.ocr_rec_min_width, math.ceil(self.rec_h * crop.shape[1] / crop.shape[0]))
        if required > self.cfg.ocr_rec_max_width:
            raise FeatureError("Text line too long; aim closer at a smaller section")
        width = self.fixed_rec_w or int(math.ceil(required/32)*32)
        if self.fixed_rec_w and required > self.fixed_rec_w:
            raise FeatureError("Fixed recognition width would squash this long line")
        return ctc_decode(infer(self.rec, normalize_line(crop, self.rec_h, width)), self.characters)

    def warmup(self):
        self.detect(np.zeros((256,320,3), np.uint8))
        # Independently warm recognition even when blank detection finds no boxes.
        self.recognize(np.full((48,192,3), 255, np.uint8))

    def read(self, image: np.ndarray, cancel: threading.Event | None = None) -> OCRResult:
        start = time.monotonic()
        boxes = self.detect(image)
        lines, rejected = [], 0
        incomplete = len(boxes) > self.cfg.ocr_max_lines or len(boxes) >= 1500
        for box in boxes[:self.cfg.ocr_max_lines]:
            if cancel and cancel.is_set():
                raise Cancelled("OCR cancelled")
            if time.monotonic() - start > self.cfg.ocr_budget_s:
                incomplete = True
                break
            try:
                text, confidence = self.recognize(crop_quad(image, box))
            except FeatureError as exc:
                if "too long" not in str(exc) and "squash" not in str(exc):
                    raise
                rejected += 1
                incomplete = True
                continue
            if text and confidence >= self.cfg.ocr_text_confidence:
                lines.append(TextLine(text, confidence, tuple(tuple(float(v) for v in p) for p in box)))
            else:
                rejected += 1
                incomplete = True
        return OCRResult(lines, incomplete, len(boxes), rejected)


def describe_text(result: OCRResult, language: str) -> str:
    if not result.lines:
        return ("साफ़ पाठ नहीं मिला। कैमरा स्थिर रखें और अच्छी रोशनी में पास लाएं।" if language == "hi"
                else "No readable text found. Hold the camera steady and move closer in good light.")
    prefix = "लिखा है। " if language == "hi" else "The text says. "
    text = prefix + "\n".join(line.text for line in result.lines)
    if result.incomplete:
        text += ("\nकुछ पाठ नहीं पढ़ पाया। छोटा हिस्सा पास से दिखाएं।" if language == "hi"
                 else "\nSome text could not be read. Show a smaller section closer to the camera.")
    return text
