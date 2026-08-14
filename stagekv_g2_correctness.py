"""Validate Qwen2.5-7B attention computation with KV head groups of 2.

The prototype keeps a standard GPU DynamicCache and changes only the SDPA
calculation granularity. Qwen's 4 repeated KV heads and 28 query heads are split
into two aligned groups: 2 KV heads and 14 query heads per group. Group outputs
are concatenated in the original head order and compared against unmodified
standard attention over prefill plus autoregressive decode steps. The report
separates strict BF16 allclose from behavior-preserving numerical equivalence,
because changing CUDA kernel shapes can change floating-point reduction order.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as functional
from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache


DEFAULT_MODEL_PATH = "/model/ModelScope/Qwen/Qwen2.5-7B-Instruct"
DEFAULT_RESULTS_DIR = "/root/stagekv/results/day3_g2_correctness"
EXPECTED_STRUCTURE = {
    "num_hidden_layers": 28,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "hidden_size": 3584,
}


def validate_structure(config: Any) -> None:
    mismatches = {
        name: {"expected": expected, "observed": int(getattr(config, name))}
        for name, expected in EXPECTED_STRUCTURE.items()
        if int(getattr(config, name)) != expected
    }
    if mismatches:
        raise RuntimeError(
            "The selected path is not Qwen2.5-7B: " + json.dumps(mismatches)
        )


def cache_bytes(cache: DynamicCache) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for key, value in zip(cache.key_cache, cache.value_cache)
        for tensor in (key, value)
        if tensor is not None
    )


def compare_tensors(
    standard: torch.Tensor,
    grouped: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    if standard.shape != grouped.shape:
        return {
            "shape_equal": False,
            "standard_shape": list(standard.shape),
            "grouped_shape": list(grouped.shape),
            "nonfinite_pattern_equal": False,
            "finite_allclose": False,
            "max_abs_error": math.inf,
            "mean_abs_error": math.inf,
            "relative_l2_error": math.inf,
            "cosine_similarity": -1.0,
            "reference_max_abs": math.nan,
        }

    standard_float = standard.detach().float()
    grouped_float = grouped.detach().float()
    standard_finite = torch.isfinite(standard_float)
    grouped_finite = torch.isfinite(grouped_float)
    nonfinite_pattern_equal = bool(torch.equal(standard_finite, grouped_finite))
    common_finite = standard_finite & grouped_finite

    if common_finite.any():
        difference = (standard_float[common_finite] - grouped_float[common_finite]).abs()
        max_abs_error = float(difference.max().item())
        mean_abs_error = float(difference.mean().item())
        reference_values = standard_float[common_finite]
        grouped_values = grouped_float[common_finite]
        error_l2 = torch.linalg.vector_norm(reference_values - grouped_values)
        reference_l2 = torch.linalg.vector_norm(reference_values)
        relative_l2_error = float(
            (error_l2 / reference_l2.clamp_min(torch.finfo(torch.float32).tiny)).item()
        )
        cosine_similarity = float(
            functional.cosine_similarity(
                reference_values.reshape(1, -1),
                grouped_values.reshape(1, -1),
                dim=1,
                eps=1e-12,
            ).item()
        )
        reference_max_abs = float(reference_values.abs().max().item())
        finite_allclose = bool(
            torch.allclose(
                reference_values,
                grouped_values,
                rtol=rtol,
                atol=atol,
            )
        )
    else:
        max_abs_error = 0.0
        mean_abs_error = 0.0
        relative_l2_error = 0.0
        cosine_similarity = 1.0
        reference_max_abs = 0.0
        finite_allclose = True

    return {
        "shape_equal": True,
        "standard_shape": list(standard.shape),
        "grouped_shape": list(grouped.shape),
        "nonfinite_pattern_equal": nonfinite_pattern_equal,
        "finite_allclose": finite_allclose,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "relative_l2_error": relative_l2_error,
        "cosine_similarity": cosine_similarity,
        "reference_max_abs": reference_max_abs,
    }


def compare_logit_distributions(
    standard: torch.Tensor,
    grouped: torch.Tensor,
) -> dict[str, Any]:
    standard_float = standard.detach().float()
    grouped_float = grouped.detach().float()
    standard_log_prob = functional.log_softmax(standard_float, dim=-1)
    grouped_log_prob = functional.log_softmax(grouped_float, dim=-1)
    standard_prob = standard_log_prob.exp()
    grouped_prob = grouped_log_prob.exp()
    total_variation = 0.5 * (standard_prob - grouped_prob).abs().sum(dim=-1)
    standard_top = torch.topk(standard_float, k=10, dim=-1)
    grouped_top = torch.topk(grouped_float, k=10, dim=-1)
    standard_top1 = int(standard_top.indices[0, 0].item())
    grouped_top1 = int(grouped_top.indices[0, 0].item())
    standard_top1_margin = float(
        (standard_top.values[0, 0] - standard_top.values[0, 1]).item()
    )
    standard_top10 = set(int(value) for value in standard_top.indices[0].tolist())
    grouped_top10 = set(int(value) for value in grouped_top.indices[0].tolist())
    symmetric_kl = 0.5 * (
        (standard_prob * (standard_log_prob - grouped_log_prob)).sum(dim=-1)
        + (grouped_prob * (grouped_log_prob - standard_log_prob)).sum(dim=-1)
    )
    return {
        "top1_equal": standard_top1 == grouped_top1,
        "standard_top1": standard_top1,
        "grouped_top1": grouped_top1,
        "standard_top1_margin": standard_top1_margin,
        "top10_overlap": len(standard_top10 & grouped_top10) / 10.0,
        "total_variation": float(total_variation.max().item()),
        "symmetric_kl": float(symmetric_kl.max().item()),
        "max_probability_abs_error": float(
            (standard_prob - grouped_prob).abs().max().item()
        ),
    }


class GroupedSdpaPatch:
    def __init__(
        self,
        *,
        query_heads: int,
        kv_heads: int,
        kv_group_size: int,
        diagnostic_rtol: float,
        diagnostic_atol: float,
    ) -> None:
        if query_heads % kv_heads != 0:
            raise ValueError("query_heads must be divisible by kv_heads")
        if kv_heads % kv_group_size != 0:
            raise ValueError("kv_heads must be divisible by kv_group_size")
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.kv_group_size = kv_group_size
        self.query_heads_per_kv = query_heads // kv_heads
        self.query_group_size = kv_group_size * self.query_heads_per_kv
        self.expected_groups_per_attention = kv_heads // kv_group_size
        self.eligible_calls = 0
        self.group_calls = 0
        self.passthrough_calls = 0
        self.call_shapes: list[dict[str, Any]] = []
        self.local_operator_checks: list[dict[str, Any]] = []
        self._checked_query_lengths: set[int] = set()
        self.diagnostic_rtol = diagnostic_rtol
        self.diagnostic_atol = diagnostic_atol
        self._original = functional.scaled_dot_product_attention

    def _expand_mask(
        self,
        attn_mask: torch.Tensor | None,
        target_query_heads: int,
    ) -> torch.Tensor | None:
        if attn_mask is None or attn_mask.ndim < 4:
            return attn_mask
        mask_heads = attn_mask.shape[-3]
        if mask_heads in (1, target_query_heads):
            return attn_mask
        if mask_heads == self.kv_heads:
            return attn_mask.repeat_interleave(self.query_heads_per_kv, dim=-3)
        raise RuntimeError(
            f"Unsupported attention-mask head dimension: {mask_heads}; "
            f"expected 1, {self.kv_heads}, or {target_query_heads}"
        )

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        is_qwen_attention = (
            query.ndim == 4
            and key.ndim == 4
            and value.ndim == 4
            and query.shape[-3] == self.query_heads
            and key.shape[-3] in (self.kv_heads, self.query_heads)
            and value.shape[-3] == key.shape[-3]
            and query.shape[-1] == key.shape[-1] == value.shape[-1]
        )
        if not is_qwen_attention:
            self.passthrough_calls += 1
            return self._original(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
                **kwargs,
            )

        if key.shape[-3] == self.kv_heads:
            key = key.repeat_interleave(self.query_heads_per_kv, dim=-3)
            value = value.repeat_interleave(self.query_heads_per_kv, dim=-3)
            enable_gqa = False

        attn_mask = self._expand_mask(attn_mask, self.query_heads)
        self.eligible_calls += 1
        if len(self.call_shapes) < 4:
            self.call_shapes.append(
                {
                    "query": list(query.shape),
                    "key": list(key.shape),
                    "value": list(value.shape),
                    "mask": list(attn_mask.shape) if attn_mask is not None else None,
                    "is_causal": bool(is_causal),
                }
            )

        outputs: list[torch.Tensor] = []
        for query_start in range(0, self.query_heads, self.query_group_size):
            query_end = query_start + self.query_group_size
            group_mask = attn_mask
            if attn_mask is not None and attn_mask.ndim >= 4 and attn_mask.shape[-3] != 1:
                group_mask = attn_mask[..., query_start:query_end, :, :]
            outputs.append(
                self._original(
                    query[..., query_start:query_end, :, :],
                    key[..., query_start:query_end, :, :],
                    value[..., query_start:query_end, :, :],
                    attn_mask=group_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                    enable_gqa=False,
                    **kwargs,
                )
            )
            self.group_calls += 1
        grouped_output = torch.cat(outputs, dim=-3)

        query_length = int(query.shape[-2])
        if query_length not in self._checked_query_lengths:
            reference_output = self._original(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=False,
                **kwargs,
            )
            comparison = compare_tensors(
                reference_output,
                grouped_output,
                rtol=self.diagnostic_rtol,
                atol=self.diagnostic_atol,
            )
            comparison.update(
                {
                    "query_length": query_length,
                    "key_value_length": int(key.shape[-2]),
                    "phase": "decode" if query_length == 1 else "prefill",
                }
            )
            self.local_operator_checks.append(comparison)
            self._checked_query_lengths.add(query_length)
            del reference_output
        return grouped_output

    @contextmanager
    def install(self) -> Iterator["GroupedSdpaPatch"]:
        original = functional.scaled_dot_product_attention
        if original is not self._original:
            raise RuntimeError("SDPA was modified before StageKV patch installation")
        functional.scaled_dot_product_attention = self
        try:
            yield self
        finally:
            functional.scaled_dot_product_attention = original


def run_autoregressive(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    decode_tokens: int,
    patch: GroupedSdpaPatch | None,
) -> dict[str, Any]:
    cache = DynamicCache()
    attention_mask = torch.ones_like(input_ids)
    next_input = input_ids
    generated_ids: list[int] = []
    step_logits: list[torch.Tensor] = []
    final_hidden: torch.Tensor | None = None

    torch.cuda.synchronize()
    start = time.perf_counter()
    context = patch.install() if patch is not None else _null_context()
    with context:
        for step in range(decode_tokens):
            with torch.inference_mode():
                output = model.model(
                    input_ids=next_input,
                    attention_mask=attention_mask,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
                final_hidden = output.last_hidden_state[:, -1, :].detach().float().cpu()
                logits = model.lm_head(output.last_hidden_state[:, -1:, :])[:, -1, :]
                next_token = logits.argmax(dim=-1, keepdim=True)
            step_logits.append(logits.detach().float().cpu())
            generated_ids.append(int(next_token.item()))
            next_input = next_token
            attention_mask = torch.ones(
                (1, input_ids.shape[1] + step + 1),
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            del logits, output
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    assert final_hidden is not None
    return {
        "generated_ids": generated_ids,
        "step_logits": step_logits,
        "final_hidden": final_hidden,
        "cache_bytes": cache_bytes(cache),
        "cache_tokens": int(cache.get_seq_length()),
        "elapsed_seconds": elapsed,
    }


@contextmanager
def _null_context() -> Iterator[None]:
    yield None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--kv-group-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--max-relative-l2", type=float, default=5e-3)
    parser.add_argument("--min-cosine-similarity", type=float, default=0.9999)
    parser.add_argument("--max-total-variation", type=float, default=1e-2)
    args = parser.parse_args()

    if args.sequence_length < 1 or args.decode_tokens < 1:
        raise ValueError("sequence-length and decode-tokens must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    validate_structure(config)
    query_heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    layers = int(config.num_hidden_layers)
    head_dim = int(
        getattr(config, "head_dim", 0) or config.hidden_size // query_heads
    )

    print(f"model={args.model_path}", flush=True)
    print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"layers={layers} query_heads={query_heads} kv_heads={kv_heads} "
        f"head_dim={head_dim} r=0 g={args.kv_group_size}",
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
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)
    input_ids = torch.randint(
        0,
        config.vocab_size,
        (1, args.sequence_length),
        generator=generator,
        device="cuda",
    )

    print("running=standard", flush=True)
    standard = run_autoregressive(model, input_ids, args.decode_tokens, None)
    patch = GroupedSdpaPatch(
        query_heads=query_heads,
        kv_heads=kv_heads,
        kv_group_size=args.kv_group_size,
        diagnostic_rtol=args.rtol,
        diagnostic_atol=args.atol,
    )
    print("running=stagekv_g2", flush=True)
    grouped = run_autoregressive(model, input_ids, args.decode_tokens, patch)

    logits_comparisons = [
        compare_tensors(s, g, rtol=args.rtol, atol=args.atol)
        for s, g in zip(standard["step_logits"], grouped["step_logits"])
    ]
    distribution_comparisons = [
        compare_logit_distributions(s, g)
        for s, g in zip(standard["step_logits"], grouped["step_logits"])
    ]
    hidden_comparison = compare_tensors(
        standard["final_hidden"],
        grouped["final_hidden"],
        rtol=args.rtol,
        atol=args.atol,
    )
    expected_attention_calls = layers * args.decode_tokens
    expected_group_calls = expected_attention_calls * patch.expected_groups_per_attention
    theoretical_cache_bytes = (
        2
        * layers
        * (args.sequence_length + args.decode_tokens - 1)
        * kv_heads
        * head_dim
        * 2
    )

    same_sequence = standard["generated_ids"] == grouped["generated_ids"]
    logits_allclose = all(
        item["shape_equal"]
        and item["nonfinite_pattern_equal"]
        and item["finite_allclose"]
        for item in logits_comparisons
    )
    hidden_allclose = (
        hidden_comparison["shape_equal"]
        and hidden_comparison["nonfinite_pattern_equal"]
        and hidden_comparison["finite_allclose"]
    )
    cache_equal = (
        standard["cache_bytes"]
        == grouped["cache_bytes"]
        == theoretical_cache_bytes
    )
    call_counts_correct = (
        patch.eligible_calls == expected_attention_calls
        and patch.group_calls == expected_group_calls
    )
    max_logits_relative_l2 = max(
        item["relative_l2_error"] for item in logits_comparisons
    )
    min_logits_cosine = min(
        item["cosine_similarity"] for item in logits_comparisons
    )
    all_step_top1_equal = all(
        item["top1_equal"] for item in distribution_comparisons
    )
    max_probability_total_variation = max(
        item["total_variation"] for item in distribution_comparisons
    )
    min_top10_overlap = min(
        item["top10_overlap"] for item in distribution_comparisons
    )
    local_operator_numerically_equivalent = (
        len(patch.local_operator_checks) == 2
        and all(
            item["shape_equal"]
            and item["nonfinite_pattern_equal"]
            and item["relative_l2_error"] <= args.max_relative_l2
            and item["cosine_similarity"] >= args.min_cosine_similarity
            for item in patch.local_operator_checks
        )
    )
    end_to_end_numerically_equivalent = (
        max_logits_relative_l2 <= args.max_relative_l2
        and min_logits_cosine >= args.min_cosine_similarity
        and hidden_comparison["relative_l2_error"] <= args.max_relative_l2
        and hidden_comparison["cosine_similarity"] >= args.min_cosine_similarity
        and max_probability_total_variation <= args.max_total_variation
        and min_top10_overlap >= 0.9
    )
    behavioral_equivalence = (
        same_sequence
        and all_step_top1_equal
        and local_operator_numerically_equivalent
        and max_probability_total_variation <= args.max_total_variation
        and min_top10_overlap >= 0.9
        and cache_equal
        and call_counts_correct
    )

    report = {
        "status": "PASS_BEHAVIORAL_EQUIVALENCE" if behavioral_equivalence else "FAIL",
        "method": "stagekv_grouped_attention_prototype",
        "cache_policy": "standard_gpu_dynamic_cache",
        "target_resident_kv_heads_r": 0,
        "implemented_resident_policy": "not_implemented_in_this_prototype",
        "observed_gpu_cached_kv_heads": kv_heads,
        "kv_group_size_g": args.kv_group_size,
        "sequence_length": args.sequence_length,
        "decode_tokens": args.decode_tokens,
        "layers": layers,
        "query_heads": query_heads,
        "kv_heads": kv_heads,
        "query_heads_per_kv_head": query_heads // kv_heads,
        "query_heads_per_group": patch.query_group_size,
        "groups_per_layer": patch.expected_groups_per_attention,
        "expected_attention_calls": expected_attention_calls,
        "observed_attention_calls": patch.eligible_calls,
        "expected_group_calls": expected_group_calls,
        "observed_group_calls": patch.group_calls,
        "passthrough_sdpa_calls": patch.passthrough_calls,
        "sample_call_shapes": patch.call_shapes,
        "local_operator_checks": patch.local_operator_checks,
        "local_operator_numerically_equivalent": local_operator_numerically_equivalent,
        "same_generated_sequence": same_sequence,
        "standard_generated_ids": standard["generated_ids"],
        "grouped_generated_ids": grouped["generated_ids"],
        "strict_bf16_all_step_logits_allclose": logits_allclose,
        "max_step_logits_abs_error": max(
            item["max_abs_error"] for item in logits_comparisons
        ),
        "mean_step_logits_abs_error": sum(
            item["mean_abs_error"] for item in logits_comparisons
        )
        / len(logits_comparisons),
        "max_step_logits_relative_l2_error": max_logits_relative_l2,
        "min_step_logits_cosine_similarity": min_logits_cosine,
        "all_step_top1_equal": all_step_top1_equal,
        "max_probability_total_variation": max_probability_total_variation,
        "max_probability_abs_error": max(
            item["max_probability_abs_error"]
            for item in distribution_comparisons
        ),
        "max_symmetric_kl": max(
            item["symmetric_kl"] for item in distribution_comparisons
        ),
        "min_top10_overlap": min_top10_overlap,
        "minimum_standard_top1_margin": min(
            item["standard_top1_margin"] for item in distribution_comparisons
        ),
        "per_step_distribution_checks": distribution_comparisons,
        "strict_bf16_final_hidden_allclose": hidden_allclose,
        "final_hidden_max_abs_error": hidden_comparison["max_abs_error"],
        "final_hidden_relative_l2_error": hidden_comparison["relative_l2_error"],
        "final_hidden_cosine_similarity": hidden_comparison["cosine_similarity"],
        "end_to_end_numerically_equivalent": end_to_end_numerically_equivalent,
        "behavioral_equivalence": behavioral_equivalence,
        "thresholds": {
            "strict_rtol": args.rtol,
            "strict_atol": args.atol,
            "max_relative_l2": args.max_relative_l2,
            "min_cosine_similarity": args.min_cosine_similarity,
            "max_total_variation": args.max_total_variation,
            "min_top10_overlap": 0.9,
        },
        "standard_cache_bytes": standard["cache_bytes"],
        "grouped_cache_bytes": grouped["cache_bytes"],
        "theoretical_cache_bytes": theoretical_cache_bytes,
        "cache_ratio": grouped["cache_bytes"] / theoretical_cache_bytes,
        "standard_elapsed_seconds": standard["elapsed_seconds"],
        "grouped_elapsed_seconds": grouped["elapsed_seconds"],
        "note": (
            "The grouped computation is mathematically equivalent, but changing "
            "CUDA SDPA shapes can change BF16 reduction order, so bitwise or strict "
            "1e-4 equality is reported separately and is not required for behavioral "
            "equivalence. Behavioral equivalence requires identical greedy decisions, "
            "close probability distributions, local operator agreement, equal cache "
            "bytes, and correct group-call counts. CPU offload, resident heads, "
            "asynchronous transfer, and performance claims are not enabled."
        ),
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "stagekv_g2_correctness.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved={report_path}")
    if not behavioral_equivalence:
        raise RuntimeError("StageKV g=2 behavioral-equivalence test failed")
    print("stagekv_g2_correctness=PASS_BEHAVIORAL_EQUIVALENCE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
