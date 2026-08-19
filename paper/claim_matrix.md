# StageKV 论文证据台账

## 写作原则

- 原始实验 CSV 保持只读。
- `paper_ready=true` 仅表示实验协议满足，不表示性能成功。
- 跨层加速主张已经冻结。
- 本文定位为持久 GPU KV 节省与延迟/传输代价研究。

## 主张矩阵

| 编号 | 可写结论                                                     | 禁止表述                            | 证据                        |
| ---- | ------------------------------------------------------------ | ----------------------------------- | --------------------------- |
| C1   | 两个模型的生成序列、Top-1、cache ratio、非阻塞 D2H 和调度门禁全部通过。 | 适用于所有模型；完全逐位等价。      | final_analysis.json         |
| C2   | 50% KV-head residency 使持久 GPU KV 降低约 50%。             | 整体显存降低 50%。                  | final_methods.csv           |
| C3   | 7B 与 3B 的跨层/双向加速比分别为 1.004416x、1.003662x。      | 跨层预取显著加速。                  | final_cross_comparisons.csv |
| C4   | 跨层与双向方法的 H2D/D2H 字节量相同。                        | 跨层预取减少 PCIe 流量。            | final_methods.csv           |
| C5   | StageKV 是 GPU KV 空间与推理延迟之间的权衡。                 | 显存节省没有性能代价。              | formal results              |
| C6   | Transfer profile 只能解释传输机制。                          | 将单次 profiling 当成正式性能结果。 | transfer_profile runs       |

## 冻结结论

StageKV 在 Qwen2.5-7B 和 Qwen2.5-3B 的 8K 推理中保持了行为和调度正确性，并降低约 50% 的持久 GPU KV。但是，跨层预取未达到预设的 1.05x 加速阈值，因此本文不主张稳定加速。