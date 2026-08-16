# StageKV / KV Cache Offload Handoff

更新时间：2026-08-14（Asia/Shanghai）
工作目录：C:\Users\ZhuanZ\Documents\Codex\2026-07-16\qin

## 1. 任务目标

本仓库用于复现并扩展 HeadInfer 思路，研究长上下文推理中的 KV Cache offload。当前实现路线是 StageKV：

1. 将部分历史 K/V 从 GPU 持久缓存卸载到 CPU pinned memory。
2. 在 GPU 上保留 r=2 个 KV heads。
3. 使用专用 CUDA stream、非阻塞 H2D/D2H 拷贝和 CUDA event 管理缓存生命周期。
4. 在计算第 L 层时，预取第 L+1 层的历史 KV，验证跨层流水调度。

当前研究问题不是单纯“能否搬到 CPU”，而是：在保持生成行为一致、降低持久 GPU KV 占用的前提下，跨层预取是否能抵消 CPU-GPU 搬运开销。

## 2. 验收标准

### 正确性和放置门禁

- Standard、Day 10 双向异步、Day 12 跨层预取生成序列一致。
- 每一步 Top-1 与 Standard 一致。
- 理论 KV bytes 与实际总 KV bytes 的比例为 1.0。
- r=2 时持久 GPU KV 约为全量的 50%，其余 KV 位于 CPU。
- 跨层调度计数精确。在 Qwen2.5-7B、4K prompt、32 generated tokens 下：
  - 层预取 28 * 31 = 868
  - L -> L+1 lookahead 27 * 31 = 837
  - Layer 0 fallback 31
  - hits 837
  - misses 31
  - slot reuse waits 866
- blocking D2H copies 必须为 0。

### 性能门禁

正式协议默认要求：2 次 warmup、至少 5 次 measured repeats、32 个生成 token、固定随机种子、轮换方法顺序、CUDA event 计时和 trial 结束后的全设备同步。

当前预注册的性能确认阈值是：

~~~text
cross_layer_over_bidirectional_speed_ratio >= 1.05
~~~

这只是性能成功标准，不是正确性标准。当前结果没有达到该阈值。

## 3. 已完成事项

- 完成 Qwen2.5-7B-Instruct 的标准 baseline 和 KV 结构测量。
- 完成 KV head/group 探针，确认 Qwen2.5-7B 结构：28 layers、28 query heads、4 KV heads、head_dim=128。
- 完成 CPU KV offload、pinned residency、phase-aware、双缓冲异步、双向异步和跨层预取原型。
- 完成跨层预取正确性和精确调度验证。
- 完成 Day 12 formal benchmark 脚本：
  outputs/stagekv_cross_layer_calibrated_benchmark.py
- 本地已执行并通过：
  - python -m py_compile outputs/stagekv_cross_layer_calibrated_benchmark.py
  - 旧 Day 11 名称/旧 sync/async 分支扫描
  - 无 GPU 的合成汇总、验证和性能决策测试
- 远程 RTX 4090 上已完成两批独立的 4K、Qwen2.5-7B、2 warmup、5 measured repeats 结果。

### 当前 Day 12 数据结论

最新一批目录：

D:\project_forpython_start20250929_selectall\headinfer\day12_cross_layer_calibrated_2

- 15/15 measured trials 成功。
- 生成序列、Top-1、跨层调度全部通过。
- 持久 GPU KV：0.2204 GiB -> 0.1102 GiB，降低约 50%。
- 跨层相对 Day 10 的稳态解码加速比：1.01217x。
- 稳态延迟变化：约 -1.20%。
- meets_confirmatory_speed_target=false。
- 跨层相对 Standard 仍约慢 57 倍。

第一批结果目录为：

D:\project_forpython_start20250929_selectall\headinfer\day12_cross_layer_calibrated

其跨层相对 Day 10 的加速比为 1.01840x。两批独立结果都在 1.01-1.02x，目前不能声称稳定的 5% 加速。

## 4. 未完成事项

按优先级排列：

1. 可选但建议完成一次 6 次轮换平衡重复，确保三种方法在三个执行位置各出现两次。
2. 完成 8K context scaling；如果结果仍无明显收益，再决定是否跑 16K。
3. 增加至少一个第二模型。当前 benchmark 的 validate_structure 固定为 Qwen2.5-7B，不能直接拿来跑 3B，需要先做模型结构参数化。
4. 增加 H2D transfer event 的独立计时。目前 h2d_event_timing_available=false，只能证明调度计数和行为正确，不能量化搬运与计算的实际重叠比例。
5. 分析并优化 PCIe/同步瓶颈，特别是每层历史 KV 的整层搬运和 CPU-ready 等待。
6. 建立 Git 仓库、提交初始版本并推送远端；当前目录没有 Git 元数据。
7. 汇总最终论文表格、图和实验协议，完成论文写作。

## 5. 文件及用途

所有源码位于 outputs/。

### 基线和早期实验

- standard_baseline.py：标准 GPU DynamicCache baseline，测量 prefill、吞吐、KV 理论/实际 bytes。
- layer_offload_4k.py：早期 4K layer offload smoke test。
- offload_correctness.py：Standard 与 Transformers layer offload correctness 对照。
- layer_offload_benchmark.py：早期 layer offload calibrated benchmark。
- pcie_bandwidth_probe.py：测量同步/非阻塞 H2D/D2H 和专用 CUDA stream 的 PCIe 行为。

### StageKV 结构和正确性链路

- stagekv_kv_head_probe.py：探测 KV head、query-to-KV 映射和分组预算。
- stagekv_g2_correctness.py：g=2 grouped attention correctness。
- stagekv_cpu_g2_correctness.py：CPU KV Cache offload 的 r=0,g=2 原型。
- stagekv_residency_correctness.py：resident KV heads 的放置正确性。
- stagekv_pinned_residency_correctness.py：pinned CPU cache 和 GPU resident heads。
- stagekv_phase_aware_correctness.py：prefill/decode phase-aware offload。
- stagekv_async_phase_aware_correctness.py：GPU staging slots 和异步双缓冲。
- stagekv_bidirectional_async_correctness.py：Day 10 双向异步 D2H/H2D cache，提供 BidirectionalAsyncPatch 和 DeferredAsyncResidentCache。
- stagekv_cross_layer_prefetch_correctness.py：Day 12 一层 lookahead 跨层预取，提供 CrossLayerPrefetchPatch。

### Calibrated benchmark

- stagekv_calibrated_benchmark.py：Day 9 calibrated benchmark。
- stagekv_bidirectional_calibrated_benchmark.py：Day 11 双向异步 calibrated benchmark。
- stagekv_cross_layer_calibrated_benchmark.py：当前主 benchmark，严格比较：
  - standard
  - stagekv_bidirectional_r2
  - stagekv_cross_layer_r2

该脚本默认使用：

~~~text
model: /model/ModelScope/Qwen/Qwen2.5-7B-Instruct
length: 4096
decode_tokens: 32
warmup_repeats: 2
repeats: 5
resident_heads: [2]
stagekv_modes: bidirectional cross_layer
~~~

脚本会输出 day12_warmup.csv、day12_raw.csv、day12_per_token.csv、day12_summary.csv、day12_comparison.csv 和 day12_manifest.json。

### 分析脚本

- outputs/day10_analysis/analyze_day10.mjs：Day 10 CSV 分析。
- outputs/day11_analysis/analyze_day11.mjs：Day 11 CSV 分析。
- outputs/day11_analysis/analyze_day12.mjs：Day 12 结果分析。

## 6. Git 状态

本次核查结果：

~~~text
git status --short --branch -> fatal: not a git repository
git log -5 -> fatal: not a git repository
~~~

- 当前分支：不可用，当前目录没有 .git。
- 最近 commit：不可用。
- 未提交改动：无法用 Git 区分；目录中的文件应整体视为未纳入版本控制的工作副本。
- 未发现 qin/.git 或父目录 Codex/.git。

接手者如果要建立版本控制，应先在确认文件清单后执行：

~~~bash
git init
git add HANDOFF.md outputs
git commit -m "chore: snapshot StageKV KV offload experiments"
~~~

不要在未确认远端地址、用户身份和分支策略前执行 git push。

## 7. 环境和必需变量

### 已验证的远程环境

来自远程实验 manifest/运行日志：

~~~text
GPU: NVIDIA GeForce RTX 4090, approximately 23.5 GiB
OS: Ubuntu 22.04
Python: 3.12
PyTorch: 2.13.0+cu132
PyTorch CUDA build: 13.2
Transformers: 4.46.3
Accelerate: 1.1.1
Model: Qwen2.5-7B-Instruct, local-only loading
~~~

### 必需配置

没有 API key、token 或其他真实密钥环境变量。必须满足：

- CUDA 可用并且 torch.cuda.is_available() 为 true。
- 本地模型目录存在，代码使用 local_files_only=True，不会自动下载模型。
- 运行目录中存在主脚本要求的 companion .py 文件，或通过 PYTHONPATH 使其可导入。

可选变量：

- CUDA_VISIBLE_DEVICES：限制可见 GPU。
- HF_HOME、TORCH_HOME、TMPDIR：控制缓存和临时目录；脚本本身不依赖密钥。
- PYTHONUNBUFFERED=1：benchmark worker 已在子进程环境中自动设置。

## 8. 已执行命令和测试结果

### 本地静态验证

~~~powershell
python -m py_compile outputs\stagekv_cross_layer_calibrated_benchmark.py
~~~

结果：PY_COMPILE=PASS。

旧引用扫描确认没有残留：Day-11、day11_、旧 sync/async benchmark 分支和已删除类型引用均未找到。

合成数据测试验证了：

- summarize() 能正确聚合三种方法。
- comparison_rows() 能生成两条 candidate-vs-standard 和一条跨层-vs-双向对比。
- validate_results() 能执行 5 measured + 2 warmup 门禁。
- meets_confirmatory_speed_target 的判断逻辑正确。

### 远程正式启动方式

~~~bash
python stagekv_cross_layer_calibrated_benchmark.py \
  --model-path /model/ModelScope/Qwen/Qwen2.5-7B-Instruct \
  --results-dir /root/stagekv/results/day12_cross_layer_calibrated \
  --lengths 4096 \
  --decode-tokens 32 \
  --warmup-repeats 2 \
  --repeats 5 \
  --resident-heads 2 \
  --stagekv-modes bidirectional cross_layer
~~~

建议下一次平衡运行使用新结果目录：

~~~bash
python stagekv_cross_layer_calibrated_benchmark.py \
  --model-path /model/ModelScope/Qwen/Qwen2.5-7B-Instruct \
  --results-dir /root/stagekv/results/day13_context_8k \
  --lengths 8192 \
  --decode-tokens 32 \
  --warmup-repeats 2 \
  --repeats 5 \
  --resident-heads 2 \
  --stagekv-modes bidirectional cross_layer
~~~

### 远程数据审查结果

Day 12 4K 两批独立结果均满足：15 measured rows、6 warmup rows、正确性门禁通过、跨层调度计数精确、blocking D2H 为 0。两批跨层相对 Day 10 的 speed ratio 分别约为 1.0184x 和 1.0122x，均低于 1.05x。

## 9. 关键设计决策

- 只把 r=2 作为正式跨层实验配置；r=0/1 已完成原型探索但不作为当前性能主张。
- Standard 使用 Transformers DynamicCache。
- StageKV 使用 CPU pinned KV、两个 layer-sized GPU staging slots、dedicated H2D/D2H streams、non-blocking copies 和 CUDA events。
- 跨层 lookahead 在当前层计算期间安排下一层历史 KV；Layer 0 没有前一层可预取，因此使用 fallback。
- per-token 只同步 compute-stream end event，避免每 token 的全设备同步把异步 D2H 串行化；每个 trial 结束再全设备同步。
- 行为正确性以 greedy generated sequence 和逐步 Top-1 与 Standard 一致为门禁；不把单次 correctness timing 当作性能结论。
- 所有模型加载均为 local-only，保证实验不会受到网络下载波动影响。

## 10. 已知问题、风险和阻塞项

1. 当前跨层相对 Day 10 只有约 1.01-1.02x，不能写成稳定的 5% 加速。
2. 两种 offload 方法相对 Standard 的稳态解码都慢很多，GPU 显存节省目前不足以抵消搬运代价。
3. h2d_event_timing_available=false：当前报告证明了调度计数和行为，但没有独立量化 H2D 与计算的实际重叠时间。
4. stagekv_cpu_g2_correctness.validate_structure() 固定检查 Qwen2.5-7B 结构，3B 或其他模型会被拒绝；模型泛化需要结构参数化和单独 correctness 门禁。
5. 目前只有 Qwen2.5-7B、RTX 4090、4K context；没有 8K/16K 或第二模型结果。
6. 远程输出 CSV/JSON 位于 D: 数据目录，不在当前仓库；接手机器需要重新获得这些文件或重新运行实验。
7. benchmark 输出文件名固定为 day12_*，即使 --results-dir 指向 Day 13 目录，文件名仍然是 day12 前缀；这是命名问题，不改变数值。
8. 当前目录没有 Git 仓库、远端或提交历史，无法安全执行 push、merge 或回滚。

## 11. 给下一个 Codex 的接手提示词

~~~text
你正在接手一个 StageKV/KV Cache offload 研究仓库。

请先阅读 HANDOFF.md 和 outputs/stagekv_cross_layer_calibrated_benchmark.py，
不要假设当前目录有 Git 历史。当前已验证的是 Qwen2.5-7B、RTX 4090、4K、r=2：
跨层调度和生成正确性通过，但相对 Day 10 的两批独立 5 次实验只有约
1.01-1.02x 加速，低于 1.05x 目标；不能宣称稳定加速。

下一步优先运行 8K context benchmark，保持 standard、
stagekv_bidirectional_r2、stagekv_cross_layer_r2 三种方法，使用 2 warmup、
5 measured repeats、32 decode tokens，并写入新的 results 目录。运行后审查
manifest、summary、comparison、raw 和 per_token，检查正确性、调度计数、
KV placement、稳态 CUDA 延迟、H2D/D2H 量和顺序方差。

如果 8K 仍无明显收益，冻结跨层“加速”主张，转为显存-延迟权衡结果，
然后参数化模型结构检查并加入第二模型。不要修改原始实验 CSV，不要虚构
Git commit，不要把 paper_ready=true 解读为性能成功；该字段只代表协议条件满足。
~~~

