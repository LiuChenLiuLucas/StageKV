import time
import psutil
import torch
from transformers import AutoModelForCausalLM

MODEL_PATH = "/model/ModelScope/Qwen/Qwen2.5-3B-Instruct"
SEQ_LEN = 4096

print("loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    attn_implementation="sdpa",
    local_files_only=True,
)
model.eval()

config = model.config
head_dim = getattr(
    config,
    "head_dim",
    config.hidden_size // config.num_attention_heads,
)

input_ids = torch.randint(
    0,
    config.vocab_size,
    (1, SEQ_LEN),
    device="cuda",
)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

start = time.perf_counter()

with torch.inference_mode():
    outputs = model.model(
        input_ids=input_ids,
        use_cache=True,
        return_dict=True,
    )

torch.cuda.synchronize()
elapsed = time.perf_counter() - start

cache = outputs.past_key_values
kv_bytes = 0

if hasattr(cache, "key_cache"):
    for key, value in zip(cache.key_cache, cache.value_cache):
        kv_bytes += key.numel() * key.element_size()
        kv_bytes += value.numel() * value.element_size()
else:
    for layer_cache in cache:
        key, value = layer_cache[:2]
        kv_bytes += key.numel() * key.element_size()
        kv_bytes += value.numel() * value.element_size()

theoretical_kv_bytes = (
    2
    * config.num_hidden_layers
    * SEQ_LEN
    * config.num_key_value_heads
    * head_dim
    * 2
)

print(f"sequence_length={SEQ_LEN}")
print(f"layers={config.num_hidden_layers}")
print(f"query_heads={config.num_attention_heads}")
print(f"kv_heads={config.num_key_value_heads}")
print(f"head_dim={head_dim}")
print(f"prefill_seconds={elapsed:.4f}")
print(f"prefill_tokens_per_second={SEQ_LEN / elapsed:.2f}")
print(
    f"peak_gpu_allocated_gib="
    f"{torch.cuda.max_memory_allocated() / 1024**3:.4f}"
)
print(
    f"peak_gpu_reserved_gib="
    f"{torch.cuda.max_memory_reserved() / 1024**3:.4f}"
)
print(f"measured_kv_gib={kv_bytes / 1024**3:.6f}")
print(f"theoretical_kv_gib={theoretical_kv_bytes / 1024**3:.6f}")
print(f"kv_ratio={kv_bytes / theoretical_kv_bytes:.6f}")
print(
    f"process_cpu_gib="
    f"{psutil.Process().memory_info().rss / 1024**3:.4f}"
)
print("smoke_test=PASS")