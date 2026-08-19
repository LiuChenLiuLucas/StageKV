# StageKV Paper Plan

## Working Title

StageKV: Characterizing the Memory-Latency Tradeoff of KV Cache Offloading for Long-Context LLM Inference

## Research Questions

- RQ1：StageKV 是否保持生成和缓存正确性？
- RQ2：StageKV 能减少多少持久 GPU KV？
- RQ3：显存节省产生多少解码延迟和传输代价？
- RQ4：跨层预取能否稳定改善双向异步卸载？

## Claimed Contributions

1. 实现 resident-head CPU KV offload、异步双向搬运和跨层预取。
2. 建立生成、Top-1、cache ratio、非阻塞传输和调度计数门禁。
3. 在两个 Qwen2.5 模型上实现约 50% 的持久 GPU KV 节省。
4. 证明跨层调度本身不足以消除 PCIe 传输代价，并给出可复现的瓶颈证据。

## Paper Structure

1. Introduction
2. Background and Related Work
3. StageKV Design
4. Experimental Methodology
5. Correctness and Memory Reduction
6. Latency and Cross-Layer Evaluation
7. Transfer-Cost Diagnosis
8. Limitations
9. Conclusion

## Required Tables and Figures

- Table 1：模型结构和实验协议。
- Table 2：正式性能、GPU/CPU KV 和正确性。
- Table 3：Cross/Bidirectional ratio 与 1.05x 阈值。
- Figure 1：StageKV CPU/GPU KV 与跨层预取流程。
- Figure 2：GPU KV 与 steady latency 的权衡图。
- Figure 3：两个模型的 speed ratio 和 1.05x 阈值线。
- Appendix：单次 transfer-event profiling。

## Limitations

- 仅测试 Qwen2.5-7B、3B 和 RTX 4090。
- 正式实验仅使用 8K context 和 5 次重复。
- 7B 存在较大的执行顺序方差。
- Transfer profile 只有单次 measured run。
- 7B formal manifest 缺少 model_structure 字段，但 profiling manifest 含完整结构。
- 当前原型没有减少 H2D/D2H 字节量。

## Completion Order

先完成实验设置和结果章节，再制作表格与图，之后撰写设计、引言和相关工作，最后根据真实结果写摘要和结论。