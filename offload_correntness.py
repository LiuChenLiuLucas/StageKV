"""Compare standard generation with Transformers' layer-offloaded cache."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, OffloadedCache


MODEL_PATH = "/root/ModelScope/model/Qwen2.5-7B-Instruct"
PROMPT = (
    "请阅读下面的问题并给出简洁、准确的回答。"
    "问题：为什么长上下文推理会导致KV Cache占用显存快速增长？"
)
MAX_NEW_TOKENS = 16


def load_model() -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model.eval()
    return model, tokenizer


def run(mode: str) -> tuple[dict, torch.Tensor]:
    model, tokenizer = load_model()
    inputs = tokenizer(PROMPT, return_tensors="pt")
    inputs = {name: value.to("cuda") for name, value in inputs.items()}

    with torch.inference_mode():
        warmup_output = model.generate(
            **inputs,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            max_new_tokens=1,
            use_cache=True,
        )
    torch.cuda.synchronize()
    del warmup_output
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    kwargs = {
        **inputs,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "max_new_tokens": MAX_NEW_TOKENS,
        "use_cache": True,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    if mode == "layer_offload":
        # Passing the cache object explicitly avoids the incompatible
        # OffloadedStaticCache constructor in some Transformers 4.46 builds.
        kwargs["past_key_values"] = OffloadedCache()

    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    sequence = output.sequences[0].detach().cpu()
    first_score = output.scores[0].detach().float().cpu()
    result = {
        "mode": mode,
        "elapsed_seconds": elapsed,
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "generated_token_ids": sequence.tolist(),
        "first_step_argmax": int(first_score.argmax(dim=-1).item()),
        "text": tokenizer.decode(sequence, skip_special_tokens=True),
    }

    del output, model, tokenizer, inputs
    gc.collect()
    torch.cuda.empty_cache()
    return result, first_score


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"model={MODEL_PATH}")

    standard, standard_score = run("standard")
    offloaded, offloaded_score = run("layer_offload")

    standard_ids = standard["generated_token_ids"]
    offloaded_ids = offloaded["generated_token_ids"]
    same_sequence = standard_ids == offloaded_ids
    same_first_token = standard["first_step_argmax"] == offloaded["first_step_argmax"]
    score_difference = (standard_score - offloaded_score).abs()
    max_abs_error = float(score_difference.max().item())
    mean_abs_error = float(score_difference.mean().item())
    logits_allclose = bool(
        torch.allclose(standard_score, offloaded_score, rtol=1e-4, atol=1e-4)
    )

    comparison = {
        "standard": standard,
        "layer_offload": offloaded,
        "same_generated_sequence": same_sequence,
        "same_first_token": same_first_token,
        "first_step_logits_allclose": logits_allclose,
        "first_step_logits_max_abs_error": max_abs_error,
        "first_step_logits_mean_abs_error": mean_abs_error,
    }

    output_path = Path("offload_correctness.json")
    output_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print(f"saved={output_path.resolve()}")
    if not same_sequence or not same_first_token:
        raise RuntimeError("Correctness check failed: generated tokens differ")
    print("correctness_test=PASS")


if __name__ == "__main__":
    main()
