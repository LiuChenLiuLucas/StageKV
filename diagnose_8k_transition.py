from __future__ import annotations

import faulthandler
import os
import signal
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path("/root/stagekv")
MODEL_PATH = Path("/model/ModelScope/Qwen/Qwen2.5-7B-Instruct")


def signal_name(return_code: int) -> str:
    signal_number = -return_code if return_code < 0 else return_code - 128
    try:
        return signal.Signals(signal_number).name
    except ValueError:
        return f"UNKNOWN_SIGNAL_{signal_number}"


def run_child(scenario: str) -> int:
    faulthandler.enable(all_threads=True)

    import gc
    import psutil
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    sys.path.insert(0, str(PROJECT_DIR))

    from stagekv_cpu_g2_correctness import validate_structure
    from stagekv_cross_layer_calibrated_benchmark import (
        MethodSpec,
        run_once,
    )

    def memory_state(label: str) -> None:
        torch.cuda.synchronize()
        gpu_free, gpu_total = torch.cuda.mem_get_info(0)
        ram = psutil.virtual_memory()

        print(
            f"[{label}] "
            f"GPU allocated={torch.cuda.memory_allocated() / 1024**3:.3f} GiB, "
            f"reserved={torch.cuda.memory_reserved() / 1024**3:.3f} GiB, "
            f"free={gpu_free / 1024**3:.3f}/{gpu_total / 1024**3:.3f} GiB, "
            f"RAM available={ram.available / 1024**3:.3f}/"
            f"{ram.total / 1024**3:.3f} GiB",
            flush=True,
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    config = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True)
    validate_structure(config)

    layers = int(config.num_hidden_layers)
    query_heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    head_dim = int(
        getattr(config, "head_dim", 0)
        or config.hidden_size // query_heads
    )

    print(f"scenario={scenario}", flush=True)
    print("loading model", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    generator = torch.Generator(device="cuda")
    generator.manual_seed(2026 + 8192)
    input_ids = torch.randint(
        0,
        config.vocab_size,
        (1, 8192),
        generator=generator,
        device="cuda",
    )

    if scenario == "standard_only":
        tests = [
            ("standard", MethodSpec("standard"), 3),
        ]
    elif scenario == "cross_then_standard":
        tests = [
            ("cross_layer_32", MethodSpec("cross_layer", 2), 32),
            ("standard_after_cross", MethodSpec("standard"), 3),
        ]
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    memory_state("model-loaded")

    for name, spec, decode_tokens in tests:
        print(
            f"\nBEGIN {name}, decode_tokens={decode_tokens}",
            flush=True,
        )
        memory_state(f"before-{name}")

        row = run_once(
            model,
            spec,
            input_ids,
            layers=layers,
            query_heads=query_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            decode_tokens=decode_tokens,
        )

        print(
            f"PASS {name}: "
            f"prefill={row['prefill_cuda_ms']:.3f} ms, "
            f"steady={row['decode_steady_cuda_mean_ms']:.3f} ms",
            flush=True,
        )

        del row
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        memory_state(f"after-{name}")

    print(f"\nSCENARIO PASS: {scenario}", flush=True)
    return 0


def run_parent() -> int:
    scenarios = ["standard_only", "cross_then_standard"]
    environment = dict(os.environ)
    environment["PYTHONFAULTHANDLER"] = "1"
    environment["TORCH_SHOW_CPP_STACKTRACES"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"

    for scenario in scenarios:
        print(f"\n{'=' * 70}")
        print(f"START CHILD SCENARIO: {scenario}")
        print(f"{'=' * 70}", flush=True)

        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child", scenario],
            cwd=PROJECT_DIR,
            env=environment,
        )

        code = completed.returncode
        print(f"\nCHILD RETURN CODE: {code}", flush=True)

        if code == 0:
            print(f"RESULT: {scenario} passed", flush=True)
            continue

        if code < 0 or code >= 128:
            print(
                f"RESULT: {scenario} terminated by "
                f"{signal_name(code)}",
                flush=True,
            )
        else:
            print(
                f"RESULT: {scenario} ended with Python/process error {code}",
                flush=True,
            )

        return code

    print("\nALL DIAGNOSTIC SCENARIOS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        raise SystemExit(run_child(sys.argv[2]))
    raise SystemExit(run_parent())