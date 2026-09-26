"""PI_AWARENESS settings for the assembled Raspberry Pi 5 hardware.

Default hardware profile:
  * Raspberry Pi Camera Rev 1.3 / Camera Module 1 through Picamera2
  * INMP441 I2S microphone on the googlevoicehat ALSA card
  * MAX98357A I2S amplifier on the same card
  * VL53L0X at I2C address 0x29

Normal runtime is fully offline.  A 32 GB microSD is storage, not RAM.
"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass
class CameraSettings:
    backend: str = "picamera2"   # fixed for the attached CSI camera
    device: int = 0
    width: int = 1280
    height: int = 960
    fps: int = 10
    rotation: int = 0
    mirror: bool = False
    autofocus: bool = False      # Camera Rev 1.3 / Module 1 is fixed-focus
    max_frame_age_s: float = 1.0
    startup_timeout_s: float = 5.0
    warmup_s: float = 0.35
    on_demand: bool = True       # sensor is opened only for a vision command


@dataclass
class VoiceSettings:
    languages: tuple[str, ...] = ("en", "hi")
    backend: str = "inmp441_i2s"     # inmp441_i2s / sounddevice (desktop fallback)
    microphone: str | int | None = None
    channels: int = 2                 # INMP441 ALSA stream is stereo-framed
    channel_index: int = 0            # L/R pin is tied to GND -> left channel
    preferred_sample_rate: int = 16000
    i2s_capture_rate: int = 48000
    i2s_gain: float = 10.0
    block_ms: int = 50
    rms_threshold: float = 3500.0      # tune with tools/mic_test.py on the assembled unit
    silence_s: float = 0.50
    min_speech_s: float = 0.15
    max_utterance_s: float = 7.0
    confidence: float = 0.58
    candidate_margin: float = 0.10
    use_grammar: bool = True
    require_wake_prefix: bool = False
    asr_en: Path = ROOT / "models/vosk-model-small-en-in-0.4"
    asr_hi: Path = ROOT / "models/vosk-model-small-hi-0.22"
    speaker_device: str | None = "i2s-auto"  # auto-resolves the shared googlevoicehat card
    english_voice: str = "en"
    hindi_voice: str = "hi"
    speech_rate: int = 165
    speech_chunk_chars: int = 150
    listen_gap_s: float = 1.5
    echo_tail_s: float = 0.35
    ready_text_en: str = "I am ready"
    ready_text_hi: str = "मैं तैयार हूँ।"


@dataclass
class VisionSettings:
    detector: str = "yolo26s"
    object_model: Path = ROOT / "models/yolo26s_640.onnx"
    object_input_size: int = 640
    object_format: str = "community_yolos"
    object_threads: int = 2
    object_confidence: float = 0.40
    object_iou: float = 0.45
    max_object_names: int = 8
    ort_threads: int = 2
    ocr_det: Path = ROOT / "models/ocr_det.onnx"
    ocr_rec: Path = ROOT / "models/ocr_rec.onnx"
    ocr_cls: Path = ROOT / "models/ocr_cls.onnx"
    ocr_use_cls: bool = True
    ocr_det_limit: int = 736
    ocr_bitmap_threshold: float = 0.30
    ocr_box_threshold: float = 0.55
    ocr_unclip_ratio: float = 1.5
    ocr_text_confidence: float = 0.62
    ocr_rec_height: int = 48
    ocr_rec_min_width: int = 320
    ocr_rec_max_width: int = 1536
    ocr_max_lines: int = 80
    ocr_budget_s: float = 8.0
    color_roi_fraction: float = 0.45
    color_min_dominance: float = 0.50
    color_constancy: bool = True


@dataclass
class DistanceSettings:
    enabled: bool = True
    driver: str = "vl53l0x"
    address: int = 0x29
    timing_budget_us: int = 200000
    samples_per_query: int = 1          # one ~200 ms measurement keeps voice latency low
    max_age_s: float = 1.0
    min_distance_mm: int = 30
    max_distance_mm: int = 1200         # conservative useful range from the supplied hardware helper
    default_step_mm: float = 750.0
    calibration_file: Path = ROOT / "state/calibration.json"
    on_demand: bool = True              # no continuous I2C polling in idle mode


@dataclass
class PowerSettings:
    idle_mode: bool = True
    camera_on_demand: bool = True
    distance_on_demand: bool = True
    preload_detector: bool = True       # one boot-time warmup; zero detector CPU while idle
    preload_ocr: bool = False           # OCR initializes on first "read" command
    unload_ocr_after_s: float = 0.0     # 0 = retain once loaded; avoids repeated CPU-heavy loads


@dataclass
class Settings:
    camera: CameraSettings = field(default_factory=CameraSettings)
    voice: VoiceSettings = field(default_factory=VoiceSettings)
    vision: VisionSettings = field(default_factory=VisionSettings)
    distance: DistanceSettings = field(default_factory=DistanceSettings)
    power: PowerSettings = field(default_factory=PowerSettings)
    response_language: str = "auto"
    preload_models: bool = False        # legacy switch; per-model PowerSettings above now controls warmup
    block_internet: bool = True
    log_dir: Path = ROOT / "logs"
    log_recognized_text: bool = True


CONFIG = Settings()
