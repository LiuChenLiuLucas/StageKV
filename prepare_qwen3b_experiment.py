from __future__ import annotations

import py_compile
import shutil
from pathlib import Path

from transformers import AutoConfig


MODEL_PATH = Path("/model/ModelScope/Qwen/Qwen2.5-3B-Instruct")
REMOTE_PROJECT_DIR = Path("/root/stagekv")
BENCHMARK_NAME = "stagekv_cross_layer_calibrated_benchmark.py"
COMPANION_FILES = (
    "stagekv_cpu_g2_correctness.py",
    "stagekv_pinned_residency_correctness.py",
    "stagekv_bidirectional_async_correctness.py",
    "stagekv_cross_layer_prefetch_correctness.py",
)
EXPECTED_STRUCTURE = {
    "model_type": "qwen2",
    "num_hidden_layers": 36,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "hidden_size": 2048,
    "head_dim": 128,
}


def main() -> None:
    source_dir = Path(__file__).resolve().parent
    source_benchmark = source_dir / BENCHMARK_NAME
    target_benchmark = REMOTE_PROJECT_DIR / BENCHMARK_NAME
    backup_benchmark = REMOTE_PROJECT_DIR / (
        "stagekv_cross_layer_calibrated_benchmark.pre_parameterized.py"
    )

    if not source_benchmark.is_file():
        raise RuntimeError(f"Missing parameterized benchmark: {source_benchmark}")
    source_text = source_benchmark.read_text(encoding="utf-8")
    if "def validate_model_structure(" not in source_text:
        raise RuntimeError("The local benchmark is not the parameterized version")
    if not REMOTE_PROJECT_DIR.is_dir():
        raise RuntimeError(f"Missing remote project directory: {REMOTE_PROJECT_DIR}")

    source_resolved = source_benchmark.resolve()
    target_resolved = target_benchmark.resolve() if target_benchmark.exists() else None
    if source_resolved != target_resolved:
        if target_benchmark.is_file() and not backup_benchmark.exists():
            shutil.copy2(target_benchmark, backup_benchmark)
            print(f"backup={backup_benchmark}")
        shutil.copy2(source_benchmark, target_benchmark)
        print(f"installed={target_benchmark}")
    else:
        print(f"benchmark_already_in_place={target_benchmark}")

    for name in COMPANION_FILES:
        path = REMOTE_PROJECT_DIR / name
        if not path.is_file():
            raise RuntimeError(f"Missing companion file: {path}")

    if not MODEL_PATH.is_dir():
        raise RuntimeError(f"Missing Qwen2.5-3B model directory: {MODEL_PATH}")
    config = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True)
    observed = {
        "model_type": str(config.model_type),
        "num_hidden_layers": int(config.num_hidden_layers),
        "num_attention_heads": int(config.num_attention_heads),
        "num_key_value_heads": int(config.num_key_value_heads),
        "hidden_size": int(config.hidden_size),
        "head_dim": int(
            getattr(config, "head_dim", 0)
            or config.hidden_size // config.num_attention_heads
        ),
    }
    if observed != EXPECTED_STRUCTURE:
        raise RuntimeError(
            f"Unexpected Qwen2.5-3B structure: expected={EXPECTED_STRUCTURE}, "
            f"observed={observed}"
        )

    py_compile.compile(str(target_benchmark), doraise=True)
    print(f"model_structure={observed}")
    print("resident_heads=1")
    print("gpu_kv_residency_fraction=0.5")
    print("prepare_qwen3b_experiment=PASS")


if __name__ == "__main__":
    main()
