from __future__ import annotations
import os
from pathlib import Path
from .common import FeatureError

def session(path: Path, threads: int = 2):
    if not Path(path).is_file():
        raise FeatureError(f"Missing model: {path}. Run python tools/download_models.py during setup.")
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise FeatureError("ONNX Runtime is not installed; run bash install.sh") from exc
    ort.disable_telemetry_events()
    opts = ort.SessionOptions()
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False  # OCR widths vary; avoid growing retained shape-specific arenas
    opts.intra_op_num_threads = max(1, min(threads, os.cpu_count() or 1))
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
    result = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
    if len(result.get_inputs()) != 1 or result.get_inputs()[0].type != "tensor(float)":
        raise FeatureError(f"Expected a single FP32 ONNX image input: {path.name}")
    return result

def infer(sess, array):
    return sess.run(None, {sess.get_inputs()[0].name: array})[0]

def fixed_dimension(value, default: int) -> int:
    return int(value) if isinstance(value, int) and value > 0 else default
