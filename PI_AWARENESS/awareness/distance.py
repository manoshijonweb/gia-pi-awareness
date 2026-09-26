from __future__ import annotations
from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
import time
from .common import atomic_json, FeatureError

@dataclass(frozen=True)
class RangeReading:
    mm: float | None
    at: float
    valid: bool
    reason: str = ""
    raw_status: int | None = None

class Calibration:
    def __init__(self, path: Path, default_step_mm: float = 750.0):
        self.path = path
        self.step_mm = self.validate(default_step_mm)
        self.calibrated = False
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.step_mm = self.validate(data["step_length_mm"])
                self.calibrated = True
            except (ValueError, TypeError, KeyError, OSError) as exc:
                logging.getLogger(__name__).warning("Invalid calibration; using default: %s", exc)

    @staticmethod
    def validate(mm: float) -> float:
        if isinstance(mm, bool):
            raise ValueError("Step length must be a number")
        mm = float(mm)
        if not math.isfinite(mm) or not 200 <= mm <= 1200:
            raise ValueError("Step length must be between 200 and 1200 mm")
        return mm

    def save(self, mm: float, source: str = "manual", **extra) -> None:
        mm = self.validate(mm)
        atomic_json(self.path, {"schema": 1, "step_length_mm": mm,
                    "updated_unix": time.time(), "source": source, **extra})
        self.step_mm = mm
        self.calibrated = True

    def from_walk(self, distance_mm: float, steps: int) -> None:
        if isinstance(steps, bool) or int(steps) != steps or steps < 2:
            raise ValueError("Use at least two counted individual steps, not strides")
        distance_mm = float(distance_mm)
        if not math.isfinite(distance_mm) or distance_mm <= 0:
            raise ValueError("Measured distance must be positive")
        self.save(distance_mm / steps, "measured_walk", measured_distance_mm=distance_mm, steps=steps)


def describe_distance(reading: RangeReading, step_mm: float, language: str = "en",
                      now: float | None = None, max_age_s: float = 0.4,
                      minimum_mm: float = 100, maximum_mm: float = 4000) -> str:
    now = time.monotonic() if now is None else now
    step_mm = Calibration.validate(step_mm)
    invalid = (not reading.valid or reading.mm is None or not math.isfinite(reading.mm)
               or reading.mm < minimum_mm or reading.mm > maximum_mm
               or not math.isfinite(reading.at) or now - reading.at > max_age_s or reading.at > now + 0.01)
    if invalid:
        return ("दूरी का भरोसेमंद माप नहीं मिला। रास्ता खाली मानकर न चलें।" if language == "hi"
                else "Distance unavailable. Do not assume the path is clear.")
    steps = reading.mm / step_mm
    if steps < 1:
        return ("सावधान। सेंसर के सामने एक कदम से कम दूरी पर वस्तु है।" if language == "hi"
                else "Caution. Object less than one step ahead of the sensor.")
    # Quantise downward to half steps: never overstate how far away an obstacle is.
    estimate = math.floor(steps * 2) / 2
    number = str(int(estimate)) if estimate.is_integer() else f"{estimate:.1f}"
    return (f"सेंसर के सामने लगभग {number} कदम पर वस्तु है।" if language == "hi"
            else f"Object about {number} steps ahead of the sensor.")
