"""Day-7 StageKV phase-aware correctness sweep for r in {0, 1, 2, 4}.

During the initial prefill, newly projected K/V tensors are already on GPU.
This prototype writes them to the configured static resident/pinned cache but
uses the original GPU tensors directly for attention, avoiding a redundant
CPU-to-GPU readback. During decode, attention reads resident heads from GPU and
offloaded heads from pinned CPU memory in groups of at most two KV heads.

The script validates behavior, cache placement, phase routing, and transfer
accounting. Transfers are not overlapped with compute, so elapsed time remains
diagnostic and is not a performance claim.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from stagekv_cpu_g2_correctness import (
    CPUHeadOffloadPatch,
    compare_logit_distributions,
    compare_tensors,
    run_autoregressive,
    validate_structure,
)
from stagekv_pinned_residency_correctness import (
    KV_GROUP_SIZE,
    RESIDENT_VALUES,
    StaticPinnedResidentCache,
    expected_transfer_groups_per_attention,
)


DEFAULT_MODEL_PATH = "/model/ModelScope/Qwen/Qwen2.5-7B-Instruct"
DEFAULT_RESULTS_DIR = "/root/stagekv/results/day7_phase_aware"


class PhaseAwareOffloadPatch(CPUHeadOffloadPatch):
    """Use fresh GPU K/V during prefill and cached K/V during decode."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        cache: StaticPinnedResidentCache,
        *,
        query_heads: int,
        kv_heads: int,
        kv_group_size: int,
    ) -> None:
        super().__init__(
            model,
            cache,
            query_heads=query_heads,
            kv_heads=kv_heads,
            kv_group_size=kv_group_size,
        )
        self.cache: StaticPinnedResidentCache
        self.prefill_attention_calls = 0
        self.decode_attention_calls = 0
        self.prefill_direct_group_calls = 0
        self.decode_cached_group_calls = 0
        self.prefill_cpu_to_gpu_group_transfers = 0
        self.decode_cpu_to_gpu_group_transfers = 0

    def _attention_forward(
        self,
        module: Any,
        layer_idx: int,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: Any | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None, StaticPinnedResidentCache]:
        del cache_position, kwargs
        if output_attentions:
            raise RuntimeError("This prototype supports output_attentions=False only")
        if not use_cache or past_key_value is not self.cache:
            raise RuntimeError(
                "PhaseAwareOffloadPatch requires its cache and use_cache=True"
            )

        batch_size, query_length, _ = hidden_states.size()
        previous_cache_length = self.cache.get_seq_length(layer_idx)
        is_prefill = previous_cache_length == 0

        query_states = module.q_proj(hidden_states)
        key_states = module.k_proj(hidden_states)
        value_states = module.v_proj(hidden_states)
        query_states = query_states.view(
            batch_size, query_length, module.num_heads, module.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, query_length, module.num_key_value_heads, module.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, query_length, module.num_key_value_heads, module.head_dim
        ).transpose(1, 2)

        if position_embeddings is None:
            if not hasattr(module, "rotary_emb"):
                raise RuntimeError(
                    "This Transformers version must pass position_embeddings"
                )
            cos, sin = module.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        # Persist K/V according to r, but keep the original GPU tensors alive
        # for direct prefill attention. Decode reads the complete cached prefix.
        self.cache.append(layer_idx, key_states, value_states)
        transfers_before_attention = self.cache.cpu_to_gpu_group_transfers
        outputs: list[torch.Tensor] = []
        for kv_start in range(0, self.kv_heads, self.kv_group_size):
            kv_end = kv_start + self.kv_group_size
            query_start = kv_start * self.query_heads_per_kv
            query_end = kv_end * self.query_heads_per_kv
            if is_prefill:
                key_group = key_states[:, kv_start:kv_end]
                value_group = value_states[:, kv_start:kv_end]
                self.prefill_direct_group_calls += 1
            else:
                key_group, value_group = self.cache.load_group(
                    layer_idx,
                    kv_start,
                    kv_end,
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                self.decode_cached_group_calls += 1

            key_group = key_group.repeat_interleave(self.query_heads_per_kv, dim=1)
            value_group = value_group.repeat_interleave(self.query_heads_per_kv, dim=1)
            group_mask = self._group_mask(attention_mask, query_start, query_end)
            is_causal = group_mask is None and query_length > 1
            outputs.append(
                functional.scaled_dot_product_attention(
                    query_states[:, query_start:query_end],
                    key_group,
                    value_group,
                    attn_mask=group_mask,
                    dropout_p=0.0,
                    is_causal=is_causal,
                )
            )
            self.group_calls += 1
            del key_group, value_group

        transfers_after_attention = self.cache.cpu_to_gpu_group_transfers
        transfer_delta = transfers_after_attention - transfers_before_attention
        if is_prefill:
            self.prefill_attention_calls += 1
            self.prefill_cpu_to_gpu_group_transfers += transfer_delta
        else:
            self.decode_attention_calls += 1
            self.decode_cpu_to_gpu_group_transfers += transfer_delta

        attn_output = torch.cat(outputs, dim=1)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(
            batch_size, query_length, module.hidden_size
        )
        attn_output = module.o_proj(attn_output)
        self.attention_calls += 1
        if len(self.call_shapes) < 4:
            self.call_shapes.append(
                {
                    "query_shape": list(query_states.shape),
                    "raw_kv_cache_shape": self.cache.cache_shape(layer_idx),
                    "previous_cache_length": previous_cache_length,
                    "query_length": query_length,
                    "phase": "prefill" if is_prefill else "decode",
                    "kv_source": "fresh_gpu" if is_prefill else "hybrid_cache",
                    "cpu_to_gpu_group_transfers": transfer_delta,
                }
            )
        return attn_output, None, self.cache


def expected_h2d_bytes(
    *,
    layers: int,
    offloaded_heads: int,
    sequence_length: int,
    decode_tokens: int,
    head_dim: int,
    element_size: int,
) -> int:
    # Step zero is prefill and performs no H2D readback. For subsequent
    # decode steps, the cached lengths are L+1, ..., L+decode_tokens-1.
    decode_steps = decode_tokens - 1
    summed_cache_lengths = (
        decode_steps * sequence_length
        + decode_steps * (decode_steps + 1) // 2
    )
    return (
        2
        * layers
        * offloaded_heads
        * summed_cache_lengths
        * head_dim
        * element_size
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--max-total-variation", type=float, default=1e-2)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.sequence_length < 1 or args.decode_tokens < 1:
        raise ValueError("sequence-length and decode-tokens must be positive")

    config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    validate_structure(config)
    layers = int(config.num_hidden_layers)
    query_heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    head_dim = int(getattr(config, "head_dim", 0) or config.hidden_size // query_heads)
    max_cache_len = args.sequence_length + args.decode_tokens - 1
    element_size = 2  # BF16
    print(f"model={args.model_path}", flush=True)
    print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"layers={layers} query_heads={query_heads} kv_heads={kv_heads} "
        f"head_dim={head_dim} max_cache_len={max_cache_len} "
        f"g={KV_GROUP_SIZE} r={RESIDENT_VALUES}",
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

    print("running=standard_gpu_dynamic_cache", flush=True)
    standard = run_autoregressive(model, input_ids, args.decode_tokens)
    total_used_bytes = 2 * layers * kv_heads * max_cache_len * head_dim * element_size
    expected_attention_calls = layers * args.decode_tokens
    expected_prefill_attention_calls = layers
    expected_decode_attention_calls = layers * (args.decode_tokens - 1)
    expected_prefill_direct_group_calls = layers * (kv_heads // KV_GROUP_SIZE)
    expected_decode_cached_group_calls = (
        expected_decode_attention_calls * (kv_heads // KV_GROUP_SIZE)
    )
    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []

    for resident_heads in RESIDENT_VALUES:
        torch.cuda.empty_cache()
        cache = StaticPinnedResidentCache(
            layers=layers,
            kv_heads=kv_heads,
            resident_heads=resident_heads,
            max_cache_len=max_cache_len,
        )
        patch = PhaseAwareOffloadPatch(
            model,
            cache,
            query_heads=query_heads,
            kv_heads=kv_heads,
            kv_group_size=KV_GROUP_SIZE,
        )
        print(f"running=stagekv_phase_aware_r{resident_heads}_g2", flush=True)
        candidate = run_autoregressive(
            model, input_ids, args.decode_tokens, cpu_patch=patch
        )
        torch.cuda.synchronize()

        logits_checks = [
            compare_tensors(a, b, rtol=args.rtol, atol=args.atol)
            for a, b in zip(standard["step_logits"], candidate["step_logits"])
        ]
        distribution_checks = [
            compare_logit_distributions(a, b)
            for a, b in zip(standard["step_logits"], candidate["step_logits"])
        ]
        max_tv = max(item["total_variation"] for item in distribution_checks)
        min_top10_overlap = min(item["top10_overlap"] for item in distribution_checks)
        expected_gpu_bytes = total_used_bytes * resident_heads // kv_heads
        expected_cpu_bytes = total_used_bytes - expected_gpu_bytes
        expected_h2d_groups = (
            expected_decode_attention_calls
            * expected_transfer_groups_per_attention(resident_heads)
        )
        expected_h2d_byte_count = expected_h2d_bytes(
            layers=layers,
            offloaded_heads=kv_heads - resident_heads,
            sequence_length=args.sequence_length,
            decode_tokens=args.decode_tokens,
            head_dim=head_dim,
            element_size=element_size,
        )
        expected_allocations = layers * 2 * int(resident_heads > 0) + layers * 2 * int(
            resident_heads < kv_heads
        )
        cache_checks_correct = (
            cache.placement_correct()
            and cache.all_cpu_buffers_pinned()
            and cache.used_gpu_bytes() == expected_gpu_bytes
            and cache.used_cpu_bytes() == expected_cpu_bytes
            and cache.allocated_gpu_bytes() == expected_gpu_bytes
            and cache.allocated_cpu_bytes() == expected_cpu_bytes
            and all(length == max_cache_len for length in cache.lengths)
            and cache.allocations == expected_allocations
            and cache.cache_growth_cat_calls == 0
        )
        phase_routing_correct = (
            patch.prefill_attention_calls == expected_prefill_attention_calls
            and patch.decode_attention_calls == expected_decode_attention_calls
            and patch.prefill_direct_group_calls == expected_prefill_direct_group_calls
            and patch.decode_cached_group_calls == expected_decode_cached_group_calls
            and patch.prefill_cpu_to_gpu_group_transfers == 0
            and patch.decode_cpu_to_gpu_group_transfers == expected_h2d_groups
        )
        transfer_counts_correct = (
            patch.attention_calls == expected_attention_calls
            and cache.gpu_to_cpu_append_calls == expected_attention_calls
            and cache.cpu_to_gpu_group_transfers == expected_h2d_groups
            and cache.non_blocking_h2d_calls == expected_h2d_groups * 2
            and cache.cpu_to_gpu_bytes == expected_h2d_byte_count
        )
        same_sequence = standard["generated_ids"] == candidate["generated_ids"]
        all_top1_equal = all(item["top1_equal"] for item in distribution_checks)
        checks = {
            "same_generated_sequence": same_sequence,
            "all_step_top1_equal": all_top1_equal,
            "probability_total_variation_within_threshold": (
                max_tv <= args.max_total_variation
            ),
            "top10_overlap_within_threshold": min_top10_overlap >= 0.9,
            "cache_checks_correct": cache_checks_correct,
            "phase_routing_correct": phase_routing_correct,
            "transfer_counts_and_bytes_correct": transfer_counts_correct,
        }
        behavioral_equivalence = all(checks.values())
        report = {
            "status": (
                "PASS_BEHAVIORAL_EQUIVALENCE" if behavioral_equivalence else "FAIL"
            ),
            "resident_kv_heads_r": resident_heads,
            "kv_group_size_g": KV_GROUP_SIZE,
            "sequence_length": args.sequence_length,
            "decode_tokens": args.decode_tokens,
            "cache_implementation": "preallocated_static_pinned_cpu",
            "phase_aware_prefill_enabled": True,
            "prefill_kv_source": "fresh_gpu",
            "decode_kv_source": "hybrid_resident_pinned_cache",
            "async_overlap_enabled": False,
            "same_generated_sequence": same_sequence,
            "all_step_top1_equal": all_top1_equal,
            "max_probability_total_variation": max_tv,
            "min_top10_overlap": min_top10_overlap,
            "max_step_logits_relative_l2_error": max(
                item["relative_l2_error"] for item in logits_checks
            ),
            "min_step_logits_cosine_similarity": min(
                item["cosine_similarity"] for item in logits_checks
            ),
            "cpu_buffers_pinned": cache.all_cpu_buffers_pinned(),
            "cache_growth_cat_calls": cache.cache_growth_cat_calls,
            "stagekv_gpu_cache_bytes": cache.used_gpu_bytes(),
            "stagekv_cpu_cache_bytes": cache.used_cpu_bytes(),
            "expected_gpu_cache_bytes": expected_gpu_bytes,
            "expected_cpu_cache_bytes": expected_cpu_bytes,
            "cache_checks_correct": cache_checks_correct,
            "expected_prefill_attention_calls": expected_prefill_attention_calls,
            "observed_prefill_attention_calls": patch.prefill_attention_calls,
            "expected_decode_attention_calls": expected_decode_attention_calls,
            "observed_decode_attention_calls": patch.decode_attention_calls,
            "expected_prefill_direct_group_calls": expected_prefill_direct_group_calls,
            "observed_prefill_direct_group_calls": patch.prefill_direct_group_calls,
            "expected_decode_cached_group_calls": expected_decode_cached_group_calls,
            "observed_decode_cached_group_calls": patch.decode_cached_group_calls,
            "prefill_cpu_to_gpu_group_transfers": (
                patch.prefill_cpu_to_gpu_group_transfers
            ),
            "expected_decode_cpu_to_gpu_group_transfers": expected_h2d_groups,
            "observed_decode_cpu_to_gpu_group_transfers": (
                patch.decode_cpu_to_gpu_group_transfers
            ),
            "phase_routing_correct": phase_routing_correct,
            "expected_cpu_to_gpu_bytes": expected_h2d_byte_count,
            "observed_cpu_to_gpu_bytes": cache.cpu_to_gpu_bytes,
            "non_blocking_h2d_tensor_copies": cache.non_blocking_h2d_calls,
            "transfer_counts_and_bytes_correct": transfer_counts_correct,
            "standard_elapsed_seconds": standard["elapsed_seconds"],
            "stagekv_phase_aware_elapsed_seconds": candidate["elapsed_seconds"],
            "behavioral_checks": checks,
            "failed_checks": [name for name, passed in checks.items() if not passed],
            "note": (
                "Prefill writes K/V to the configured cache but evaluates attention "
                "from the fresh GPU K/V tensors, so prefill H2D readback must be zero. "
                "Decode uses the hybrid cache. No asynchronous transfer stream or "
                "compute/transfer overlap is enabled; timing is diagnostic only."
            ),
        }
        reports.append(report)
        rows.append(report.copy())
        del candidate, cache, patch

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "stagekv_phase_aware.json"
    csv_path = results_dir / "stagekv_phase_aware.csv"
    document = {
        "model": args.model_path,
        "performance_claim_enabled": False,
        "results": reports,
    }
    json_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(document, ensure_ascii=False, indent=2))
    print(f"saved={json_path}")
    print(f"saved={csv_path}")
    if not all(item["status"] == "PASS_BEHAVIORAL_EQUIVALENCE" for item in reports):
        raise RuntimeError("At least one phase-aware residency case failed")
    print("stagekv_phase_aware=PASS_ALL_R_VALUES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
