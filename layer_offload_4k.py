"""4K correctness and resource check for Transformers layer KV offload.

This is a one-run validation experiment, not the final repeated benchmark.
It compares greedy generation from an identical 4096-token input using:
1. the default GPU-resident KV cache; and
2. Transformers OffloadedCache, which keeps layer caches in CPU memory.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import psutil
import torch
from transformers import AutoModelForCausalLM, OffloadedCache


MODEL_PATH = "/root/ModelScope/model/Qwen2.5-7B-Instruct"
RESULT_PATH = Path("/root/headinfer/results/layer_offload_4k.json")
SEQUENCE_LENGTH = 4096
MAX_NEW_TOKENS = 4
SEED = 2026


def load_model() -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model.eval()
    return model


def generate_once(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mode: str,
) -> tuple[dict[str, Any], torch.Tensor]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "min_new_tokens": MAX_NEW_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "use_cache": True,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    if mode == "layer_offload":
        kwargs["past_key_values"] = OffloadedCache()

    process = psutil.Process()
    cpu_rss_before = process.memory_info().rss
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    cpu_rss_after = process.memory_info().rss

    generated_ids = output.sequences[0, SEQUENCE_LENGTH:].detach().cpu()
    first_step_logits = output.scores[0].detach().float().cpu()
    result = {
        "mode": mode,
        "input_tokens": SEQUENCE_LENGTH,
        "generated_tokens": int(generated_ids.numel()),
        "generated_token_ids": generated_ids.tolist(),
        "elapsed_seconds": elapsed,
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "cpu_rss_before_gib": cpu_rss_before / 1024**3,
        "cpu_rss_after_gib": cpu_rss_after / 1024**3,
        "cpu_rss_delta_gib": (cpu_rss_after - cpu_rss_before) / 1024**3,
        "first_step_argmax": int(first_step_logits.argmax(dim=-1).item()),
    }

    del output
    gc.collect()
    torch.cuda.empty_cache()
    return result, first_step_logits


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"model={MODEL_PATH}")
    print(f"sequence_length={SEQUENCE_LENGTH}")

    model = load_model()
    generator = torch.Generator(device="cuda")
    generator.manual_seed(SEED)
    input_ids = torch.randint(
        low=0,
        high=model.config.vocab_size,
        size=(1, SEQUENCE_LENGTH),
        generator=generator,
        device="cuda",
    )
    attention_mask = torch.ones_like(input_ids)

    # Short warm-up initializes CUDA kernels without creating a 4K cache.
    warmup_ids = input_ids[:, :128]
    warmup_mask = attention_mask[:, :128]
    with torch.inference_mode():
        warmup_output = model.generate(
            input_ids=warmup_ids,
            attention_mask=warmup_mask,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            min_new_tokens=1,
            max_new_tokens=1,
            use_cache=True,
        )
    del warmup_output
    gc.collect()
    torch.cuda.empty_cache()

    print("running=standard", flush=True)
    standard, standard_logits = generate_once(
        model, input_ids, attention_mask, "standard"
    )
    print("running=layer_offload", flush=True)
    offloaded, offloaded_logits = generate_once(
        model, input_ids, attention_mask, "layer_offload"
    )

    difference = (standard_logits - offloaded_logits).abs()
    same_sequence = standard["generated_token_ids"] == offloaded["generated_token_ids"]
    same_first_token = standard["first_step_argmax"] == offloaded["first_step_argmax"]
    logits_allclose = bool(
        torch.allclose(standard_logits, offloaded_logits, rtol=1e-4, atol=1e-4)
    )

    result = {
        "model": MODEL_PATH,
        "seed": SEED,
        "sequence_length": SEQUENCE_LENGTH,
        "max_new_tokens": MAX_NEW_TOKENS,
        "standard": standard,
        "layer_offload": offloaded,
        "same_generated_sequence": same_sequence,
        "same_first_token": same_first_token,
        "first_step_logits_allclose": logits_allclose,
        "first_step_logits_max_abs_error": float(difference.max().item()),
        "first_step_logits_mean_abs_error": float(difference.mean().item()),
        "offload_time_overhead_percent": (
            offloaded["elapsed_seconds"] / standard["elapsed_seconds"] - 1
        )
        * 100,
        "allocated_gpu_saving_gib": (
            standard["peak_gpu_allocated_gib"]
            - offloaded["peak_gpu_allocated_gib"]
        ),
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved={RESULT_PATH}")

    if not same_sequence or not same_first_token:
        raise RuntimeError("4K correctness check failed: generated tokens differ")
    print("layer_offload_4k=PASS")


if __name__ == "__main__":
    main()
