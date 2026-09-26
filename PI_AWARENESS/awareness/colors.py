"""Central ROI color naming, circular-HSV k-means (three clusters).
Neutral-reference gains are a lightweight heuristic; this is NOT the learned
CVPR 2019 Quasi-Unsupervised Color Constancy method or a calibrated colorimeter.
"""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from .common import FeatureError

HI_COLORS = {"black": "काला", "white": "सफ़ेद", "gray": "स्लेटी", "red": "लाल",
             "orange": "नारंगी", "yellow": "पीला", "green": "हरा", "cyan": "फ़िरोज़ी",
             "blue": "नीला", "purple": "बैंगनी", "pink": "गुलाबी", "brown": "भूरा"}

@dataclass(frozen=True)
class ColorResult:
    name: str
    qualifier: str
    dominance: float
    lighting_ok: bool
    correction_applied: bool


def correct_illuminant(image: np.ndarray) -> tuple[np.ndarray, bool]:
    # Estimate from low-saturation reference pixels in the WHOLE view. Do not
    # apply naive gray-world to an all-red central object and turn it gray.
    sample = cv2.resize(image, (96, 72))
    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] < 38) & (hsv[:, :, 2] > 60) & (hsv[:, :, 2] < 240)
    if mask.sum() < sample.shape[0] * sample.shape[1] * 0.03:
        return image, False
    means = np.median(sample[mask].astype(np.float32), axis=0)
    gain = np.clip(means.mean() / np.maximum(means, 1), 0.85, 1.18)
    corrected = np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return corrected, bool(np.max(np.abs(gain - 1)) > 0.01)


def hsv_name(h: float, s: float, v: float) -> tuple[str, str]:
    if v < 38:
        return "black", ""
    if s < 35:
        return ("white", "") if v >= 205 else ("gray", "dark" if v < 95 else "")
    if 5 <= h < 23 and v < 165:
        return "brown", "dark" if v < 85 else ""
    if (h < 10 or h >= 165) and s < 170 and v >= 170:
        return "pink", ""
    if h < 10 or h >= 170:
        name = "red"
    elif h < 23:
        name = "orange"
    elif h < 36:
        name = "yellow"
    elif h < 86:
        name = "green"
    elif h < 101:
        name = "cyan"
    elif h < 131:
        name = "blue"
    else:
        name = "purple" if h < 162 else "pink"
    qualifier = "dark" if v < 125 else ("light" if s < 110 and v > 185 else "")
    return name, qualifier


def identify_color(image: np.ndarray, cfg) -> ColorResult:
    if image is None or image.ndim != 3 or image.shape[2] != 3 or image.size < 48:
        raise FeatureError("No usable color image")
    corrected, applied = correct_illuminant(image) if cfg.color_constancy else (image, False)
    h, w = corrected.shape[:2]
    fraction = max(0.1, min(1.0, cfg.color_roi_fraction))
    rh, rw = max(2, int(h * fraction)), max(2, int(w * fraction))
    top, left = (h - rh) // 2, (w - rw) // 2
    roi = cv2.resize(corrected[top:top + rh, left:left + rw], (80, 80))
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    # Hue lies on a circle, so red around 0/179 must be close, not opposite.
    angle = hsv[:, 0] * (2 * np.pi / 180)
    saturation = hsv[:, 1] / 255
    feature = np.stack((np.cos(angle) * saturation, np.sin(angle) * saturation,
                        hsv[:, 2] / 255), axis=1).astype(np.float32)
    cv2.setRNGSeed(19)
    _, labels, _ = cv2.kmeans(feature, 3, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.01), 2, cv2.KMEANS_PP_CENTERS)
    counts: dict[tuple[str, str], int] = {}
    pixels = roi.reshape(-1, 3)
    for cluster in range(3):
        selected = labels.ravel() == cluster
        if not selected.any():
            continue
        median = np.median(pixels[selected], axis=0).astype(np.uint8)
        hue, sat, val = cv2.cvtColor(median.reshape(1, 1, 3), cv2.COLOR_BGR2HSV).reshape(3)
        key = hsv_name(float(hue), float(sat), float(val))
        counts[key] = counts.get(key, 0) + int(selected.sum())
    winner = max(counts, key=counts.get)
    brightness = float(np.median(hsv[:, 2]))
    return ColorResult(winner[0], winner[1], counts[winner] / len(hsv), brightness >= 22, applied)


def describe_color(result: ColorResult, language: str, min_dominance: float = 0.5) -> str:
    if not result.lighting_ok:
        return "रंग पहचानने के लिए रोशनी कम है।" if language == "hi" else "Too dark to identify the color reliably."
    if result.dominance < min_dominance:
        return ("बीच में कई रंग हैं। वस्तु को और पास रखें।" if language == "hi"
                else "Several colors in the center. Hold the object closer.")
    if language == "hi":
        prefix = {"dark": "गहरा ", "light": "हल्का ", "": ""}[result.qualifier]
        return f"बीच का मुख्य रंग {prefix}{HI_COLORS[result.name]} है।"
    name = (result.qualifier + " " if result.qualifier else "") + result.name
    return f"The dominant color in the center is {name}."
