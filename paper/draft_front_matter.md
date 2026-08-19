# Conclusion

本文实现并评估了 StageKV，一种基于 KV-head residency 的细粒度 KV Cache offloading 原型。

在 Qwen2.5-3B 和 Qwen2.5-7B 的 8192-token 推理实验中，StageKV 在保持生成序列、逐步 Top-1、cache placement、非阻塞 D2H 和跨层调度正确的前提下，将 persistent GPU KV Cache 减少约 50%。

实验同时显示，显存节省伴随着明显的延迟和数据搬运代价。7B 模型的 Cross-layer 延迟约为 Standard 的 113.80 倍，3B 模型约为 Standard 的 1.41 倍。Cross-layer 相对于 Bidirectional 的正式速度比接近 1.0x，未达到 1.05x 的确认目标。因此，当前结果不支持“跨层预取稳定加速”的结论。

StageKV 的主要价值是提供一种可验证的 GPU KV 空间压缩机制，并揭示 KV offloading 中传输、同步和 staging buffer 等因素的影响。未来工作应包括传统 layer-wise offload 基线、其他模型架构、更多上下文长度，以及能够根据显存压力和传输成本动态选择驻留比例的自适应策略。