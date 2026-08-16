# StageKV Final 8K Analysis

This report is derived from existing CSV/JSON results. Original experiment files are unchanged.

## Method Results

| Model | Run | Method | Trials | Steady CUDA ms | Std ms | GPU KV GiB | CPU KV GiB | H2D GiB | D2H GiB | H2D event ms | D2H event ms | Order spread ms | Correctness | Schedule |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| qwen7b | formal | stagekv_bidirectional_r2 | 5 | 2943.486 | 564.009 | 0.219578 | 0.219578 | 6.793667 | 0.219578 | 0.000 | 0.000 | 914.964 | True | True |
| qwen7b | formal | stagekv_cross_layer_r2 | 5 | 2930.545 | 432.424 | 0.219578 | 0.219578 | 6.793667 | 0.219578 | 0.000 | 0.000 | 416.016 | True | True |
| qwen7b | formal | standard | 5 | 25.751 | 0.231 | 0.439156 | 0.000000 | 0.000000 | 0.000000 | 0.000 | 0.000 | 0.239 | True | True |
| qwen3b | formal | stagekv_bidirectional_r1 | 5 | 34.596 | 0.075 | 0.141157 | 0.141157 | 4.367357 | 0.141157 | 0.000 | 0.000 | 0.116 | True | True |
| qwen3b | formal | stagekv_cross_layer_r1 | 5 | 34.470 | 0.232 | 0.141157 | 0.141157 | 4.367357 | 0.141157 | 0.000 | 0.000 | 0.391 | True | True |
| qwen3b | formal | standard | 5 | 24.405 | 0.168 | 0.282314 | 0.000000 | 0.000000 | 0.000000 | 0.000 | 0.000 | 0.285 | True | True |
| qwen7b | transfer_profile | stagekv_bidirectional_r2 | 1 | 2629.440 | 0.000 | 0.219578 | 0.219578 | 6.793667 | 0.219578 | 74139.863 | 2044.113 | 0.000 | True | True |
| qwen7b | transfer_profile | stagekv_cross_layer_r2 | 1 | 3104.493 | 0.000 | 0.219578 | 0.219578 | 6.793667 | 0.219578 | 88855.661 | 3912.195 | 0.000 | True | True |
| qwen7b | transfer_profile | standard | 1 | 25.631 | 0.000 | 0.439156 | 0.000000 | 0.000000 | 0.000000 | 0.000 | 0.000 | 0.000 | True | True |
| qwen3b | transfer_profile | stagekv_bidirectional_r1 | 1 | 35.538 | 0.000 | 0.141157 | 0.141157 | 4.367357 | 0.141157 | 204.378 | 77.415 | 0.000 | True | True |
| qwen3b | transfer_profile | stagekv_cross_layer_r1 | 1 | 35.127 | 0.000 | 0.141157 | 0.141157 | 4.367357 | 0.141157 | 206.915 | 77.898 | 0.000 | True | True |
| qwen3b | transfer_profile | standard | 1 | 23.718 | 0.000 | 0.282314 | 0.000000 | 0.000000 | 0.000000 | 0.000 | 0.000 | 0.000 | True | True |

## Cross-Layer Comparison

| Model | Run | Cross/Bidirectional | Target 1.05x | Bidirectional H2D event ms | Cross H2D event ms | Bidirectional D2H event ms | Cross D2H event ms |
|---|---|---:|---|---:|---:|---:|---:|
| qwen7b | formal | 1.004416 | False | 0.000 | 0.000 | 0.000 | 0.000 |
| qwen3b | formal | 1.003662 | False | 0.000 | 0.000 | 0.000 | 0.000 |
| qwen7b | transfer_profile | 0.846979 | False | 74139.863 | 88855.661 | 2044.113 | 3912.195 |
| qwen3b | transfer_profile | 1.011726 | False | 204.378 | 206.915 | 77.415 | 77.898 |

## Gate Summary

- `qwen7b/formal`: all gates pass = `True`, structure metadata present = `False`, paper_ready = `False`
- `qwen3b/formal`: all gates pass = `True`, structure metadata present = `True`, paper_ready = `True`
- `qwen7b/transfer_profile`: all gates pass = `True`, structure metadata present = `True`, paper_ready = `False`
- `qwen3b/transfer_profile`: all gates pass = `True`, structure metadata present = `True`, paper_ready = `False`

## Decision

> Freeze the cross-layer acceleration claim. Report StageKV as a persistent GPU-KV memory reduction with a latency and transfer-cost tradeoff; `paper_ready` is protocol status only and does not imply performance success.
