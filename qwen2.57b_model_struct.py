from transformers import AutoConfig

path = "/root/ModelScope/Qwen/Qwen2.5-7B-Instruct"
c = AutoConfig.from_pretrained(path, local_files_only=True)

print("model_type:", c.model_type)
print("layers:", c.num_hidden_layers)
print("query_heads:", c.num_attention_heads)
print("kv_heads:", c.num_key_value_heads)
print("hidden_size:", c.hidden_size)
print("head_dim:", getattr(c, "head_dim", c.hidden_size // c.num_attention_heads))
print("cmax_position_embeddings:", c.max_position_embeddings)