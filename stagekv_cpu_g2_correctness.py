"""Validate synchronous CPU KV-head offload for Qwen2.5-7B (r=0, g=2).

This is a correctness-first StageKV prototype.  It changes Qwen's SDPA
attention at the point where K/V still have their original four GQA heads:

* each layer's K/V cache is stored only on CPU;
* two KV heads are copied back to GPU at a time;
* their corresponding 14 query heads are evaluated with SDPA; and
* the two attention outputs are concatenated in the original head order.

The implementation is intentionally synchronous and does not claim a speedup.
It is only suitable for batch size one and Qwen2.5-7B with SDPA attention.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Any, Iterator

import torch
import torch.nn.functional as functional
from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache
from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb


DEFAULT_MODEL_PATH = "/model/ModelScope/Qwen/Qwen2.5-7B-Instruct"
DEFAULT_RESULTS_DIR = "/root/stagekv/results/day4_cpu_g2_correctness"
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


def dynamic_cache_bytes(cache: DynamicCache) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for key, value in zip(cache.key_cache, cache.value_cache)
        for tensor in (key, value)
        if tensor is not None
    )


def compare_tensors(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        return {
            "shape_equal": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "nonfinite_pattern_equal": False,
            "finite_allclose": False,
            "max_abs_error": math.inf,
            "mean_abs_error": math.inf,
            "relative_l2_error": math.inf,
            "cosine_similarity": -1.0,
        }

    reference_float = reference.detach().float()
    candidate_float = candidate.detach().float()
    reference_finite = torch.isfinite(reference_float)
    candidate_finite = torch.isfinite(candidate_float)
    common_finite = reference_finite & candidate_finite
    nonfinite_pattern_equal = bool(torch.equal(reference_finite, candidate_finite))
    if not common_finite.any():
        return {
            "shape_equal": True,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "nonfinite_pattern_equal": nonfinite_pattern_equal,
            "finite_allclose": True,
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "relative_l2_error": 0.0,
            "cosine_similarity": 1.0,
        }

    reference_values = reference_float[common_finite]
    candidate_values = candidate_float[common_finite]
    difference = (reference_values - candidate_values).abs()
    error_l2 = torch.linalg.vector_norm(reference_values - candidate_values)
    reference_l2 = torch.linalg.vector_norm(reference_values)
    return {
        "shape_equal": True,
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "nonfinite_pattern_equal": nonfinite_pattern_equal,
        "finite_allclose": bool(
            torch.allclose(reference_values, candidate_values, rtol=rtol, atol=atol)
        ),
        "max_abs_error": float(difference.max().item()),
        "mean_abs_error": float(difference.mean().item()),
        "relative_l2_error": float(
            (error_l2 / reference_l2.clamp_min(torch.finfo(torch.float32).tiny)).item()
        ),
        "cosine_similarity": float(
            functional.cosine_similarity(
                reference_values.reshape(1, -1),
                candidate_values.reshape(1, -1),
                dim=1,
                eps=1e-12,
            ).item()
        ),
    }


def compare_logit_distributions(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, Any]:
    reference_float = reference.detach().float()
    candidate_float = candidate.detach().float()
    reference_log_prob = functional.log_softmax(reference_float, dim=-1)
    candidate_log_prob = functional.log_softmax(candidate_float, dim=-1)
    reference_prob = reference_log_prob.exp()
    candidate_prob = candidate_log_prob.exp()
    reference_top = torch.topk(reference_float, k=10, dim=-1)
    candidate_top = torch.topk(candidate_float, k=10, dim=-1)
    reference_top10 = set(int(token) for token in reference_top.indices[0].tolist())
    candidate_top10 = set(int(token) for token in candidate_top.indices[0].tolist())
    symmetric_kl = 0.5 * (
        (reference_prob * (reference_log_prob - candidate_log_prob)).sum(dim=-1)
        + (candidate_prob * (candidate_log_prob - reference_log_prob)).sum(dim=-1)
    )
    return {
        "top1_equal": int(reference_top.indices[0, 0])
        == int(candidate_top.indices[0, 0]),
        "reference_top1": int(reference_top.indices[0, 0]),
        "candidate_top1": int(candidate_top.indices[0, 0]),
        "reference_top1_margin": float(
            (reference_top.values[0, 0] - reference_top.values[0, 1]).item()
        ),
        "top10_overlap": len(reference_top10 & candidate_top10) / 10.0,
        "total_variation": float(
            (0.5 * (reference_prob - candidate_prob).abs().sum(dim=-1)).max().item()
        ),
        "symmetric_kl": float(symmetric_kl.max().item()),
        "max_probability_abs_error": float(
            (reference_prob - candidate_prob).abs().max().item()
        ),
    }


class CPUHeadCache(Cache):
    """A minimal cache used by the patched Qwen attention modules.

    Each entry has shape [batch, original_kv_heads, sequence, head_dim] and
    remains on CPU.  This class deliberately does not subclass DynamicCache:
    the patched attention path, not Transformers' normal cache update path,
    owns all cache reads and writes.
    """

    def __init__(self, layers: int, kv_heads: int) -> None:
        super().__init__()
        self.key_cache: list[torch.Tensor | None] = [None] * layers
        self.value_cache: list[torch.Tensor | None] = [None] * layers
        self.layers = layers
        self.kv_heads = kv_heads
        self.gpu_to_cpu_append_calls = 0
        self.gpu_to_cpu_bytes = 0
        self.cpu_to_gpu_group_transfers = 0
        self.cpu_to_gpu_bytes = 0

    def get_seq_length(self, layer_idx: int = 0) -> int:
        key = self.key_cache[layer_idx]
        return 0 if key is None else int(key.shape[-2])

    def get_max_cache_shape(self) -> None:
        return None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        self.append(layer_idx, key_states, value_states)
        key = self.key_cache[layer_idx]
        value = self.value_cache[layer_idx]
        assert key is not None and value is not None
        return key, value

    def append(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> None:
        if key.device.type != "cuda" or value.device.type != "cuda":
            raise RuntimeError("CPUHeadCache expects newly computed K/V on CUDA")
        key_cpu = key.detach().to(device="cpu", copy=True)
        value_cpu = value.detach().to(device="cpu", copy=True)
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_cpu
            self.value_cache[layer_idx] = value_cpu
        else:
            self.key_cache[layer_idx] = torch.cat(
                (self.key_cache[layer_idx], key_cpu), dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                (self.value_cache[layer_idx], value_cpu), dim=-2
            )
        self.gpu_to_cpu_append_calls += 1
        self.gpu_to_cpu_bytes += key_cpu.numel() * key_cpu.element_size()
        self.gpu_to_cpu_bytes += value_cpu.numel() * value_cpu.element_size()

    def load_group(
        self,
        layer_idx: int,
        start_head: int,
        end_head: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = self.key_cache[layer_idx]
        value = self.value_cache[layer_idx]
        if key is None or value is None:
            raise RuntimeError(f"Layer {layer_idx} was read before its K/V was cached")
        key_group = key[:, start_head:end_head].to(device=device, dtype=dtype)
        value_group = value[:, start_head:end_head].to(device=device, dtype=dtype)
        self.cpu_to_gpu_group_transfers += 1
        self.cpu_to_gpu_bytes += (
            key_group.numel() * key_group.element_size()
            + value_group.numel() * value_group.element_size()
        )
        return key_group, value_group

    def cpu_cache_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for key, value in zip(self.key_cache, self.value_cache)
            for tensor in (key, value)
            if tensor is not None
        )

    def cache_shape(self, layer_idx: int) -> list[int]:
        key = self.key_cache[layer_idx]
        if key is None:
            raise RuntimeError(f"Layer {layer_idx} cache is not initialized")
        return list(key.shape)

    def all_cache_tensors_on_cpu(self) -> bool:
        return all(
            tensor is None or tensor.device.type == "cpu"
            for key, value in zip(self.key_cache, self.value_cache)
            for tensor in (key, value)
        )


class CPUHeadOffloadPatch:
    """Replace Qwen SDPA attention with synchronous r=0, g=2 CPU offload."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        cache: CPUHeadCache,
        *,
        query_heads: int,
        kv_heads: int,
        kv_group_size: int,
    ) -> None:
        if query_heads % kv_heads != 0 or kv_heads % kv_group_size != 0:
            raise ValueError("Incompatible Qwen GQA heads or KV group size")
        self.cache = cache
        self.modules = [layer.self_attn for layer in model.model.layers]
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.kv_group_size = kv_group_size
        self.query_heads_per_kv = query_heads // kv_heads
        self.query_group_size = kv_group_size * self.query_heads_per_kv
        self.groups_per_attention = kv_heads // kv_group_size
        self.attention_calls = 0
        self.group_calls = 0
        self.call_shapes: list[dict[str, Any]] = []
        self._original_forwards: list[Any] = []

    @staticmethod
    def _group_mask(
        attention_mask: torch.Tensor | None,
        query_start: int,
        query_end: int,
    ) -> torch.Tensor | None:
        if attention_mask is None or attention_mask.ndim < 4:
            return attention_mask
        # A one-head causal mask broadcasts naturally.  A head-specific mask
        # must track the selected query-head range.
        if attention_mask.shape[-3] == 1:
            return attention_mask
        return attention_mask[..., query_start:query_end, :, :]

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
    ) -> tuple[torch.Tensor, None, CPUHeadCache]:
        del cache_position, kwargs
        if output_attentions:
            raise RuntimeError("This correctness prototype supports output_attentions=False only")
        if not use_cache or past_key_value is not self.cache:
            raise RuntimeError("CPUHeadOffloadPatch requires its CPUHeadCache and use_cache=True")

        batch_size, query_length, _ = hidden_states.size()
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
                    "This Transformers version must pass position_embeddings to Qwen attention"
                )
            cos, sin = module.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
        self.cache.append(layer_idx, key_states, value_states)

        outputs: list[torch.Tensor] = []
        for kv_start in range(0, self.kv_heads, self.kv_group_size):
            kv_end = kv_start + self.kv_group_size
            query_start = kv_start * self.query_heads_per_kv
            query_end = kv_end * self.query_heads_per_kv
            key_group, value_group = self.cache.load_group(
                layer_idx,
                kv_start,
                kv_end,
                device=query_states.device,
                dtype=query_states.dtype,
            )
            # Qwen's repeat_kv is exactly this repeat_interleave for each GQA head.
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
                    "query_length": query_length,
                    "phase": "decode" if query_length == 1 else "prefill",
                }
            )
        return attn_output, None, self.cache

    @contextmanager
    def install(self) -> Iterator["CPUHeadOffloadPatch"]:
        self._original_forwards = [module.forward for module in self.modules]
        def make_replacement(bound_layer_idx: int) -> Any:
            patch = self

            def replacement(
                attn_module: Any,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                position_ids: torch.Tensor | None = None,
                past_key_value: Any | None = None,
                output_attentions: bool = False,
                use_cache: bool = False,
                cache_position: torch.Tensor | None = None,
                position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
                **kwargs: Any,
            ) -> tuple[torch.Tensor, None, CPUHeadCache]:
                return patch._attention_forward(
                    attn_module,
                    bound_layer_idx,
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            return replacement

        for layer_idx, module in enumerate(self.modules):
            module.forward = MethodType(make_replacement(layer_idx), module)
        try:
            yield self
        finally:
            for module, original_forward in zip(self.modules, self._original_forwards):
                module.forward = original_forward


@contextmanager
def null_context() -> Iterator[None]:
    yield None


def run_autoregressive(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    decode_tokens: int,
    *,
    cpu_patch: CPUHeadOffloadPatch | None = None,
) -> dict[str, Any]:
    cache: DynamicCache | CPUHeadCache
    cache = cpu_patch.cache if cpu_patch is not None else DynamicCache()
    attention_mask = torch.ones_like(input_ids)
    next_input = input_ids
    generated_ids: list[int] = []
    step_logits: list[torch.Tensor] = []
    final_hidden: torch.Tensor | None = None
    context = cpu_patch.install() if cpu_patch is not None else null_context()

    torch.cuda.synchronize()
    start = time.perf_counter()
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
            generated_ids.append(int(next_token.item()))
            step_logits.append(logits.detach().float().cpu())
            next_input = next_token
            attention_mask = torch.ones(
                (1, input_ids.shape[1] + step + 1),
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            del output, logits
    torch.cuda.synchronize()
    assert final_hidden is not None
    report: dict[str, Any] = {
        "generated_ids": generated_ids,
        "step_logits": step_logits,
        "final_hidden": final_hidden,
        "cache_tokens": int(cache.get_seq_length()),
        "elapsed_seconds": time.perf_counter() - start,
    }
    if isinstance(cache, DynamicCache):
        report["cache_bytes"] = dynamic_cache_bytes(cache)
    else:
        report["cache_bytes"] = cache.cpu_cache_bytes()
    return report


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
    parser.add_argument("--max-local-relative-l2", type=float, default=5e-3)
    parser.add_argument("--min-local-cosine", type=float, default=0.9999)
    parser.add_argument("--max-total-variation", type=float, default=1e-2)
    args = parser.parse_args()
    if args.sequence_length < 1 or args.decode_tokens < 1:
        raise ValueError("sequence-length and decode-tokens must be positive")
    if args.kv_group_size != 2:
        raise ValueError("Day 4 validates only the planned r=0, g=2 prototype")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    validate_structure(config)
    layers = int(config.num_hidden_layers)
    query_heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    head_dim = int(getattr(config, "head_dim", 0) or config.hidden_size // query_heads)
    print(f"model={args.model_path}")
    print(f"gpu={torch.cuda.get_device_name(0)}")
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

    print("running=standard_gpu_dynamic_cache", flush=True)
    standard = run_autoregressive(model, input_ids, args.decode_tokens)
    torch.cuda.empty_cache()
    cpu_cache = CPUHeadCache(layers, kv_heads)
    patch = CPUHeadOffloadPatch(
        model,
        cpu_cache,
        query_heads=query_heads,
        kv_heads=kv_heads,
        kv_group_size=args.kv_group_size,
    )
    print("running=stagekv_cpu_r0_g2_sync", flush=True)
    stagekv = run_autoregressive(
        model,
        input_ids,
        args.decode_tokens,
        cpu_patch=patch,
    )

    logits_checks = [
        compare_tensors(a, b, rtol=args.rtol, atol=args.atol)
        for a, b in zip(standard["step_logits"], stagekv["step_logits"])
    ]
    distribution_checks = [
        compare_logit_distributions(a, b)
        for a, b in zip(standard["step_logits"], stagekv["step_logits"])
    ]
    hidden_check = compare_tensors(
        standard["final_hidden"], stagekv["final_hidden"], rtol=args.rtol, atol=args.atol
    )
    expected_tokens = args.sequence_length + args.decode_tokens - 1
    theoretical_cache_bytes = 2 * layers * kv_heads * expected_tokens * head_dim * 2
    expected_attention_calls = layers * args.decode_tokens
    expected_group_transfers = expected_attention_calls * patch.groups_per_attention
    same_sequence = standard["generated_ids"] == stagekv["generated_ids"]
    all_top1_equal = all(check["top1_equal"] for check in distribution_checks)
    max_total_variation = max(check["total_variation"] for check in distribution_checks)
    min_top10_overlap = min(check["top10_overlap"] for check in distribution_checks)
    max_logits_relative_l2 = max(
        check["relative_l2_error"] for check in logits_checks
    )
    min_logits_cosine = min(
        check["cosine_similarity"] for check in logits_checks
    )
    logits_nonfinite_patterns_equal = all(
        check["nonfinite_pattern_equal"] for check in logits_checks
    )
    placement_correct = (
        cpu_cache.all_cache_tensors_on_cpu()
        and cpu_cache.cpu_cache_bytes() == theoretical_cache_bytes
        and stagekv["cache_tokens"] == expected_tokens
    )
    transfer_counts_correct = (
        patch.attention_calls == expected_attention_calls
        and patch.group_calls == expected_group_transfers
        and cpu_cache.cpu_to_gpu_group_transfers == expected_group_transfers
        and cpu_cache.gpu_to_cpu_append_calls == expected_attention_calls
    )
    end_to_end_numerically_equivalent = (
        logits_nonfinite_patterns_equal
        and max_logits_relative_l2 <= args.max_local_relative_l2
        and min_logits_cosine >= args.min_local_cosine
        and hidden_check["nonfinite_pattern_equal"]
        and hidden_check["relative_l2_error"] <= args.max_local_relative_l2
        and hidden_check["cosine_similarity"] >= args.min_local_cosine
    )
    behavioral_checks = {
        "same_generated_sequence": same_sequence,
        "all_step_top1_equal": all_top1_equal,
        "logits_nonfinite_patterns_equal": logits_nonfinite_patterns_equal,
        "probability_total_variation_within_threshold": (
            max_total_variation <= args.max_total_variation
        ),
        "top10_overlap_within_threshold": min_top10_overlap >= 0.9,
        "cache_placement_correct": placement_correct,
        "transfer_counts_correct": transfer_counts_correct,
    }
    failed_checks = [
        name for name, passed in behavioral_checks.items() if not passed
    ]
    behavioral_equivalence = (
        same_sequence
        and all_top1_equal
        and logits_nonfinite_patterns_equal
        and max_total_variation <= args.max_total_variation
        and min_top10_overlap >= 0.9
        and placement_correct
        and transfer_counts_correct
    )

    report = {
        "status": "PASS_BEHAVIORAL_EQUIVALENCE" if behavioral_equivalence else "FAIL",
        "method": "stagekv_sync_cpu_kv_head_offload_prototype",
        "cache_policy": "r0_all_kv_heads_cpu_g2_sync_transfer",
        "performance_claim_enabled": False,
        "sequence_length": args.sequence_length,
        "decode_tokens": args.decode_tokens,
        "layers": layers,
        "query_heads": query_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "target_resident_kv_heads_r": 0,
        "observed_persistent_gpu_cached_kv_heads": 0,
        "kv_group_size_g": args.kv_group_size,
        "query_heads_per_kv_head": patch.query_heads_per_kv,
        "query_heads_per_group": patch.query_group_size,
        "groups_per_layer": patch.groups_per_attention,
        "same_generated_sequence": same_sequence,
        "standard_generated_ids": standard["generated_ids"],
        "stagekv_generated_ids": stagekv["generated_ids"],
        "all_step_top1_equal": all_top1_equal,
        "max_probability_total_variation": max_total_variation,
        "max_probability_abs_error": max(
            check["max_probability_abs_error"] for check in distribution_checks
        ),
        "max_symmetric_kl": max(check["symmetric_kl"] for check in distribution_checks),
        "min_top10_overlap": min_top10_overlap,
        "per_step_distribution_checks": distribution_checks,
        "strict_bf16_all_step_logits_allclose": all(
            check["shape_equal"]
            and check["nonfinite_pattern_equal"]
            and check["finite_allclose"]
            for check in logits_checks
        ),
        "max_step_logits_abs_error": max(check["max_abs_error"] for check in logits_checks),
        "max_step_logits_relative_l2_error": max_logits_relative_l2,
        "min_step_logits_cosine_similarity": min_logits_cosine,
        "strict_bf16_final_hidden_allclose": bool(
            hidden_check["shape_equal"]
            and hidden_check["nonfinite_pattern_equal"]
            and hidden_check["finite_allclose"]
        ),
        "final_hidden_max_abs_error": hidden_check["max_abs_error"],
        "final_hidden_relative_l2_error": hidden_check["relative_l2_error"],
        "final_hidden_cosine_similarity": hidden_check["cosine_similarity"],
        "end_to_end_numerically_equivalent": end_to_end_numerically_equivalent,
        "behavioral_equivalence": behavioral_equivalence,
        "behavioral_checks": behavioral_checks,
        "failed_checks": failed_checks,
        "standard_gpu_cache_bytes": standard["cache_bytes"],
        "stagekv_cpu_cache_bytes": cpu_cache.cpu_cache_bytes(),
        "stagekv_persistent_gpu_cache_bytes": 0,
        "theoretical_cache_bytes": theoretical_cache_bytes,
        "cache_placement_correct": placement_correct,
        "expected_attention_calls": expected_attention_calls,
        "observed_attention_calls": patch.attention_calls,
        "expected_cpu_to_gpu_group_transfers": expected_group_transfers,
        "observed_cpu_to_gpu_group_transfers": cpu_cache.cpu_to_gpu_group_transfers,
        "observed_gpu_to_cpu_append_calls": cpu_cache.gpu_to_cpu_append_calls,
        "gpu_to_cpu_append_bytes": cpu_cache.gpu_to_cpu_bytes,
        "cpu_to_gpu_group_bytes": cpu_cache.cpu_to_gpu_bytes,
        "transfer_counts_correct": transfer_counts_correct,
        "sample_attention_shapes": patch.call_shapes,
        "standard_elapsed_seconds": standard["elapsed_seconds"],
        "stagekv_sync_elapsed_seconds": stagekv["elapsed_seconds"],
        "thresholds": {
            "strict_rtol": args.rtol,
            "strict_atol": args.atol,
            "max_local_relative_l2": args.max_local_relative_l2,
            "min_local_cosine": args.min_local_cosine,
            "max_total_variation": args.max_total_variation,
            "min_top10_overlap": 0.9,
        },
        "note": (
            "This Day-4 prototype stores raw [B, 4, S, D] K/V tensors on CPU and "
            "synchronously transfers two KV heads per attention group. It validates "
            "correctness and placement only. The elapsed time is diagnostic, not a "
            "performance claim: pinned memory, asynchronous streams, ping-pong "
            "buffers, and resident GPU heads are not implemented."
        ),
    }
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "stagekv_cpu_g2_correctness.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved={report_path}")
    if not behavioral_equivalence:
        raise RuntimeError("StageKV CPU r=0, g=2 correctness test failed")
    print("stagekv_cpu_g2_correctness=PASS_BEHAVIORAL_EQUIVALENCE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
