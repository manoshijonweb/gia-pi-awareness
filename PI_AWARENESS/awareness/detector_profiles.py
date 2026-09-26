"""Explicit detector choices. Selecting a profile never downloads or falls back."""
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DetectorProfile:
    detector: str
    filename: str
    input_size: int
    output_format: str

PROFILES = {
    "yolo26m": DetectorProfile("yolo26m", "yolo26m_640.onnx", 640, "community_yolos"),
    "yolo26s": DetectorProfile("yolo26s", "yolo26s_640.onnx", 640, "community_yolos"),
    "yolo26n": DetectorProfile("yolo26n", "yolo26n_640.onnx", 640, "community_yolos"),
    "yolox_nano": DetectorProfile("yolox_nano", "yolox_nano.onnx", 416, "yolox_raw"),
}

def apply_profile(cfg, name: str, model_dir: Path | None = None):
    """Mutate only detector settings, not audio, camera, OCR, or step calibration."""
    if name not in PROFILES:
        raise ValueError(f"Unknown detector profile {name!r}; choose {', '.join(PROFILES)}")
    profile = PROFILES[name]
    directory = model_dir or Path(__file__).resolve().parents[1] / "models"
    cfg.detector = profile.detector
    cfg.object_model = directory / profile.filename
    cfg.object_input_size = profile.input_size
    cfg.object_format = profile.output_format
    return cfg
