"""Inspect Qwen2.5-7B GQA and per-KV-head cache layout for StageKV.

This probe does not modify attention, offload cache, or generated outputs. It
loads the model with a standard DynamicCache, records the first layer's Q/K/V
projection shapes, validates every layer's cache layout, and writes auditable
JSON/CSV artifacts for the StageKV resident-head and head-group prototypes.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache


DEFAULT_MODEL_PATH = "/model/ModelScope/Qwen/Qwen2.5-7B-Instruct"
DEFAULT_RESULTS_DIR = "/root/stagekv/results/day3_kv_head_probe"
EXPECTED_QWEN25_7B = {
    "num_hidden_layers": 28,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "hidden_size": 3584,
}


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def mib(value: int | float) -> float:
    return float(value) / 1024**2


def gib(value: int | float) -> float:
    return float(value) / 1024**3


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_model_files(model_path: str) -> None:
    model_dir = Path(model_path)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = model_dir / index_name
        if not index_path.is_file():
            continue
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shards = sorted(set(index.get("weight_map", {}).values()))
        missing = [name for name in shards if not (model_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Model download is incomplete; missing shards: {missing}"
            )
        return

    if not any(
        (model_dir / name).is_file()
        for name in ("model.safetensors", "pytorch_model.bin")
    ):
        raise FileNotFoundError(f"No model weights found in {model_path}")


def config_value(config: Any, name: str) -> int:
    value = getattr(config, name, None)
    if value is None:
        raise RuntimeError(f"Model config has no {name}")
    return int(value)


def validate_qwen25_7b(config: Any, allow_other_model: bool) -> None:
    observed = {
        name: config_value(config, name) for name in EXPECTED_QWEN25_7B
    }
    mismatches = {
        name: {"expected": expected, "observed": observed[name]}
        for name, expected in EXPECTED_QWEN25_7B.items()
        if observed[name] != expected
    }
    if mismatches and not allow_other_model:
        raise RuntimeError(
            "The selected path is not Qwen2.5-7B. "
            f"Structure mismatches: {json.dumps(mismatches)}. "
            "Pass --allow-other-model only when intentionally probing another model."
        )


def projection_metadata(module: Any) -> dict[str, int]:
    required = ("q_proj", "k_proj", "v_proj")
    if not all(hasattr(module, name) for name in required):
        raise RuntimeError("First attention module has no q_proj/k_proj/v_proj")
    return {
        "q_proj_in_features": int(module.q_proj.in_features),
        "q_proj_out_features": int(module.q_proj.out_features),
        "k_proj_in_features": int(module.k_proj.in_features),
        "k_proj_out_features": int(module.k_proj.out_features),
        "v_proj_in_features": int(module.v_proj.in_features),
        "v_proj_out_features": int(module.v_proj.out_features),
    }


def register_shape_hooks(attention: Any) -> tuple[dict[str, list[int]], list[Any]]:
    shapes: dict[str, list[int]] = {}
    handles = []

    def capture(name: str):
        def hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
            shapes[name] = list(output.shape)

        return hook

    for name in ("q_proj", "k_proj", "v_proj"):
        handles.append(getattr(attention, name).register_forward_hook(capture(name)))
    return shapes, handles


def build_query_kv_mapping(
    query_heads: int, kv_heads: int
) -> list[dict[str, int]]:
    if query_heads % kv_heads != 0:
        raise RuntimeError(
            f"Query heads ({query_heads}) must be divisible by KV heads ({kv_heads})"
        )
    query_heads_per_kv = query_heads // kv_heads
    return [
        {
            "query_head": query_head,
            "kv_head": query_head // query_heads_per_kv,
            "query_heads_per_kv_head": query_heads_per_kv,
        }
        for query_head in range(query_heads)
    ]


def inspect_cache(
    cache: DynamicCache,
    layers: int,
    kv_heads: int,
    sequence_length: int,
    head_dim: int,
) -> tuple[list[dict[str, Any]], int]:
    if len(cache.key_cache) != layers or len(cache.value_cache) != layers:
        raise RuntimeError(
            f"Expected {layers} cache layers, found "
            f"K={len(cache.key_cache)}, V={len(cache.value_cache)}"
        )

    expected_shape = [1, kv_heads, sequence_length, head_dim]
    rows: list[dict[str, Any]] = []
    total_bytes = 0
    for layer_index, (key, value) in enumerate(
        zip(cache.key_cache, cache.value_cache)
    ):
        key_shape = list(key.shape)
        value_shape = list(value.shape)
        if key_shape != expected_shape or value_shape != expected_shape:
            raise RuntimeError(
                f"Layer {layer_index} cache shape mismatch: "
                f"K={key_shape}, V={value_shape}, expected={expected_shape}"
            )
        key_bytes = tensor_bytes(key)
        value_bytes = tensor_bytes(value)
        layer_bytes = key_bytes + value_bytes
        per_head_bytes = layer_bytes // kv_heads
        total_bytes += layer_bytes
        rows.append(
            {
                "layer": layer_index,
                "key_shape": json.dumps(key_shape),
                "value_shape": json.dumps(value_shape),
                "dtype": str(key.dtype),
                "device": str(key.device),
                "kv_heads": kv_heads,
                "tokens": sequence_length,
                "head_dim": head_dim,
                "key_mib": mib(key_bytes),
                "value_mib": mib(value_bytes),
                "layer_kv_mib": mib(layer_bytes),
                "one_kv_head_mib": mib(per_head_bytes),
            }
        )
    return rows, total_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--allow-other-model", action="store_true")
    args = parser.parse_args()

    if args.sequence_length < 1:
        raise ValueError("sequence-length must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    validate_model_files(args.model_path)
    config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    validate_qwen25_7b(config, args.allow_other_model)

    layers = config_value(config, "num_hidden_layers")
    query_heads = config_value(config, "num_attention_heads")
    kv_heads = config_value(config, "num_key_value_heads")
    hidden_size = config_value(config, "hidden_size")
    if hidden_size % query_heads != 0:
        raise RuntimeError("hidden_size must be divisible by query_heads")
    head_dim = int(getattr(config, "head_dim", 0) or hidden_size // query_heads)
    query_heads_per_kv = query_heads // kv_heads

    print(f"model={args.model_path}", flush=True)
    print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"layers={layers} query_heads={query_heads} kv_heads={kv_heads} "
        f"head_dim={head_dim} query_heads_per_kv={query_heads_per_kv}",
        flush=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model.eval()

    first_attention = model.model.layers[0].self_attn
    projection_info = projection_metadata(first_attention)
    expected_projection_info = {
        "q_proj_in_features": hidden_size,
        "q_proj_out_features": query_heads * head_dim,
        "k_proj_in_features": hidden_size,
        "k_proj_out_features": kv_heads * head_dim,
        "v_proj_in_features": hidden_size,
        "v_proj_out_features": kv_heads * head_dim,
    }
    if projection_info != expected_projection_info:
        raise RuntimeError(
            f"Projection dimensions differ: observed={projection_info}, "
            f"expected={expected_projection_info}"
        )

    projection_shapes, handles = register_shape_hooks(first_attention)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)
    input_ids = torch.randint(
        0,
        config.vocab_size,
        (1, args.sequence_length),
        generator=generator,
        device="cuda",
    )
    attention_mask = torch.ones_like(input_ids)
    cache = DynamicCache()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        output = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
    torch.cuda.synchronize()
    for handle in handles:
        handle.remove()

    expected_projection_shapes = {
        "q_proj": [1, args.sequence_length, query_heads * head_dim],
        "k_proj": [1, args.sequence_length, kv_heads * head_dim],
        "v_proj": [1, args.sequence_length, kv_heads * head_dim],
    }
    if projection_shapes != expected_projection_shapes:
        raise RuntimeError(
            f"Projection output shapes differ: observed={projection_shapes}, "
            f"expected={expected_projection_shapes}"
        )

    layer_rows, measured_cache_bytes = inspect_cache(
        cache,
        layers,
        kv_heads,
        args.sequence_length,
        head_dim,
    )
    bytes_per_element = cache.key_cache[0].element_size()
    theoretical_cache_bytes = (
        2
        * layers
        * args.sequence_length
        * kv_heads
        * head_dim
        * bytes_per_element
    )
    if measured_cache_bytes != theoretical_cache_bytes:
        raise RuntimeError(
            f"KV-cache bytes differ: measured={measured_cache_bytes}, "
            f"theoretical={theoretical_cache_bytes}"
        )

    mapping_rows = build_query_kv_mapping(query_heads, kv_heads)
    one_head_all_layers_bytes = measured_cache_bytes // kv_heads
    one_head_one_layer_bytes = one_head_all_layers_bytes // layers
    group_budget_rows = [
        {
            "resident_kv_heads_r": resident_heads,
            "group_size_g": group_size,
            "resident_cache_all_layers_gib": gib(
                one_head_all_layers_bytes * resident_heads
            ),
            "offloaded_cache_all_layers_gib": gib(
                measured_cache_bytes - one_head_all_layers_bytes * resident_heads
            ),
            "one_layer_transfer_group_mib": mib(
                one_head_one_layer_bytes * group_size
            ),
            "resident_fraction_percent": resident_heads / kv_heads * 100.0,
        }
        for resident_heads in (0, 1, 2, 4)
        for group_size in (1, 2, 4)
        if resident_heads <= kv_heads and group_size <= kv_heads
    ]

    summary = {
        "status": "PASS",
        "model_path": args.model_path,
        "model_type": config.model_type,
        "sequence_length": args.sequence_length,
        "layers": layers,
        "query_heads": query_heads,
        "kv_heads": kv_heads,
        "query_heads_per_kv_head": query_heads_per_kv,
        "hidden_size": hidden_size,
        "head_dim": head_dim,
        "dtype": str(cache.key_cache[0].dtype),
        "bytes_per_element": bytes_per_element,
        "projection_metadata": projection_info,
        "projection_output_shapes": projection_shapes,
        "cache_shape_per_layer": list(cache.key_cache[0].shape),
        "measured_cache_gib": gib(measured_cache_bytes),
        "theoretical_cache_gib": gib(theoretical_cache_bytes),
        "cache_ratio": measured_cache_bytes / theoretical_cache_bytes,
        "one_kv_head_one_layer_mib": layer_rows[0]["one_kv_head_mib"],
        "one_kv_head_all_layers_gib": gib(one_head_all_layers_bytes),
        "peak_gpu_allocated_gib": gib(torch.cuda.max_memory_allocated()),
        "torch_version": torch.__version__,
        "torch_cuda_version": str(torch.version.cuda),
        "transformers_version": transformers.__version__,
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "kv_head_probe.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(results_dir / "kv_head_layers.csv", layer_rows)
    write_csv(results_dir / "query_to_kv_head.csv", mapping_rows)
    write_csv(results_dir / "stagekv_group_budgets.csv", group_budget_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary={summary_path}")
    print(f"layers_csv={results_dir / 'kv_head_layers.csv'}")
    print(f"mapping_csv={results_dir / 'query_to_kv_head.csv'}")
    print(f"budgets_csv={results_dir / 'stagekv_group_budgets.csv'}")
    print("stagekv_kv_head_probe=PASS")

    del output, cache, attention_mask, input_ids, model
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
