# vLLM 源码学习计划（2026-09 更新版）

> 目标：吃透 vLLM 推理引擎核心机制，建立从 API 入口到 GPU Kernel 的完整认知链路；
> 最终产出 = **Go LLM 网关 + 本地 vLLM 部署**整合的可运行系统（本地部署的 LLM 服务，具备 OpenAI 兼容接口、流式输出、鉴权限流等基本能力）。
> 代码基线：fork 已同步至 upstream main（2026-09-29）。**V0 引擎已整体移除**（`vllm/core/` 不复存在，`vllm/engine/` 仅剩 arg_utils/protocol 兼容壳），本计划只讲 V1 架构（`vllm/v1/`），不再做 V0 对比主线。

---

## 0. 当前状态与环境策略

### 个人基线
- Go LLM 网关骨架已完成（中间件洋葱、超时控制、并发限制、反向代理转发）。
- Java 高并发 + 6.824 分布式背景；Transformer/KV cache 理论按学习清单 Week 2 补齐中。

### 双机分工

| 机器 | 用途 |
|---|---|
| MacBook M3 Pro（本机，无 CUDA） | 源码阅读、笔记输出、Go 网关开发、压测客户端（`benchmark_serving.py` 可以从 Mac 打 WSL 的服务） |
| Windows 主机 + WSL2 + RTX 4080 24GB / 32GB RAM | vLLM 部署与压测、断点调试、CUDA kernel 实验 |

### 24GB 显存的模型梯度（由小到大）
1. **Qwen3-0.6B / 1.7B**：快速迭代专用——改代码、加日志、断点调试时秒级启动，学习期主力。
2. **Qwen3-8B（FP16 ~16GB）**：压测与正式演示主力，能稳定跑出 continuous batching 的吞吐曲线。
3. **Qwen3-14B/32B（AWQ/GPTQ INT4）**：量化专题时体验精度/显存/速度 trade-off。

### WSL2 部署备忘
```bash
# WSL2 Ubuntu 内（不要用系统 python3 / pip）：
uv venv --python 3.12 && source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto   # 源码可读可改、C++ 扩展用预编译
vllm serve Qwen/Qwen3-8B --gpu-memory-utilization 0.9
# 局域网暴露给 Mac 网关/压测：加 --host 0.0.0.0，WSL2 注意 Windows 防火墙与端口转发（netsh portproxy 或 mirrored 网络模式）
```

---

## 1. 学习原则

1. **先主线后支线**：先打通「请求进来 → 调度 → 推理 → 返回」完整链路，再深入模块细节。
2. **跑在代码前面**：每个模块先用 0.6B 小模型跑起来 + `VLLM_LOGGING_LEVEL=DEBUG` 看日志，再读源码对照。
3. **调试即学习**：关键路径打断点/加日志跟踪真实执行流，不满足于「读懂了」。
4. **输出倒逼输入**：每学完一个模块，在 `notes/` 下输出一篇笔记；核心概念必须能脱稿讲（对应学习清单的面试自测清单）。
5. **网关联动**：每个阶段都有一个「网关 × vLLM」集成任务，把源码理解立刻沉淀为最终项目的积木。

---

## Phase 0：跑通与手感（第 1 周）—— 先建立直觉，再谈源码

**目标**：在 4080 上把 vLLM 跑起来，并用已有网关完成第一次端到端串联。

- [ ] WSL2 环境搭建（见上文备忘），`vllm serve Qwen/Qwen3-0.6B` 启动成功
- [ ] 用 OpenAI SDK / curl 调 `/v1/chat/completions`，体验 SSE 流式输出（`stream=true`）
- [ ] **压测初体验**：`benchmarks/benchmark_serving.py`，并发 1/4/16/64 各测一轮，记录 TTFT/TPOT/Throughput，画出「并发-延迟-吞吐」曲线并解释拐点（学习清单 Week 3 任务）
- [ ] 换 8B 模型重复压测，观察显存占用（`nvidia-smi` 里 KV cache 占了多少）
- [ ] **集成任务 0**：Go 网关作为反向代理转发到 `http://<wsl-ip>:8000`，SSE 流式透传打通（网关的 `http.Client` + flush 语义是关键点）

**产出物**：
- `notes/00_hands_on.md` — 部署记录 + 第一份压测数据（简历素材的起点）

---

## Phase 1：入口层 —— 请求是怎么进来的（第 2 周）

**目标**：掌握 OpenAI 兼容服务的完整请求处理链。这是你网关的「对端」，值得逐行读。

> 注意：2026 年的入口层已重构，旧教程里的 `serving_chat.py` 单文件结构已不存在。

### 1.1 项目结构总览
- [ ] 目录职责速览：`vllm/`（Python 主体）、`csrc/`（CUDA/C++ kernel）、`benchmarks/`、`examples/`、`tests/`
- [ ] `vllm/entrypoints/cli/serve.py`：`vllm serve` 命令如何启动服务
- [ ] `vllm/entrypoints/openai/api_server.py`：FastAPI app 装配、路由注册、中间件挂载

### 1.2 OpenAI 兼容层（重点，与你的网关直接对应）
- [ ] `vllm/entrypoints/openai/chat_completion/`：chat 请求的 protocol（pydantic 模型）、`serving.py`（处理逻辑）、`api_router.py`
- [ ] `vllm/entrypoints/openai/completion/`：文本补全接口
- [ ] `vllm/entrypoints/serve/middleware/`：vLLM 自己实现了哪些「网关层」能力（与你 Go 网关的中间件对照阅读！）
- [ ] 流式 vs 非流式的差异处理：SSE 事件构造、usage 统计、finish_reason
- [ ] `vllm/renderers/`、`vllm/parser/`：chat template 渲染与输入解析
- [ ] `vllm/sampling_params.py`：采样参数全解析
- [ ] `vllm/entrypoints/llm.py`：离线推理 `LLM` 类（绕过 HTTP 的纯 Python API，调试时好用）

### 1.3 其他入口形态（了解即可）
- [ ] `vllm/entrypoints/anthropic/`：Anthropic API 兼容
- [ ] `vllm/entrypoints/grpc_server.py`：gRPC 服务
- [ ] `vllm/entrypoints/openai/responses/`：OpenAI Responses API

**产出物**：
- `notes/01_architecture_overview.md` — 架构总览（手绘分层图：API 层 → Engine 层 → Worker 层 → Kernel 层）
- `notes/02_api_entrypoints.md` — 入口层解析，**附「vLLM serve/middleware vs 我的 Go 网关中间件」对照表**
- **集成任务 1**：网关按 model 字段路由到不同 vLLM 实例（起 0.6B + 8B 两个实例），实现多模型路由雏形

---

## Phase 2：V1 Engine 核心 —— 调度与 KV Cache（第 3-5 周，核心中的核心）

**目标**：这是 vLLM 的灵魂，也是面试必问区（PagedAttention / continuous batching 都在这里落地）。

### 2.1 Engine 主控与进程模型
- [ ] `vllm/v1/engine/async_llm.py`：异步引擎封装，HTTP 层与引擎的接缝
- [ ] `vllm/v1/engine/core.py`：`EngineCore` 事件循环——`add_request` / `step` / 输出回传
- [ ] `vllm/v1/engine/core_client.py`：进程间通信（默认 EngineCore 跑在子进程，ZMQ IPC）
- [ ] `vllm/v1/engine/llm_engine.py`：同步引擎
- [ ] `vllm/v1/engine/admission_control.py`：**新增模块**，准入控制——对比你网关的并发限制信号量，这是引擎侧的「限流」
- [ ] `vllm/v1/engine/coordinator.py`：DP 协调（单卡场景了解概念即可）

### 2.2 调度器 Scheduler（continuous batching 本体）
- [ ] `vllm/v1/core/sched/scheduler.py`：逐行精读
  - waiting / running 队列如何流转
  - 每个 step 如何决定执行哪些请求（chunked prefill：prefill 和 decode 混在一个 batch）
  - 抢占（preemption）：显存不够时谁被踢、踢到哪、怎么恢复
- [ ] 动手验证：压测时开 DEBUG 日志观察调度决策，把 Phase 0 的吞吐曲线和调度行为对应起来

### 2.3 KV Cache 管理（PagedAttention 的工程实现）
- [ ] `vllm/v1/core/kv_cache_manager.py`：KV block 的分配/释放/复用入口
- [ ] `vllm/v1/core/block_pool.py`：物理 block 池（对应 OS 的页框管理）
- [ ] `vllm/v1/core/kv_cache_utils.py`：block 哈希与前缀缓存匹配
- [ ] `vllm/v1/core/kv_cache_coordinator.py`、`single_type_kv_cache_manager.py`：多类型 KV（全注意力/滑动窗口/MLA）如何统一协调
- [ ] **理论对齐**：PagedAttention 论文（SOSP'23）此时精读，对着代码讲「逻辑 block → 物理 block 映射」
- [ ] 亲手算一遍：Qwen3-8B、batch 32、seq 4K 需要多少 KV 显存，和 `--gpu-memory-utilization` 的实际占用对账

### 2.4 请求生命周期收尾
- [ ] `vllm/v1/request.py`：请求状态机
- [ ] `vllm/v1/engine/input_processor.py` → `detokenizer.py` → `output_processor.py`：tokenize → 增量 detokenize → stop 条件
- [ ] `vllm/v1/metrics/`：引擎侧指标（Phase 0 压测看到的 TTFT/TPOT 在哪里被统计）

**产出物**：
- `notes/03_engine_core.md` — Engine 事件循环与进程模型（含时序图）
- `notes/04_scheduler.md` — continuous batching 与抢占，**能脱稿讲**
- `notes/05_kv_cache.md` — PagedAttention 工程实现，**能脱稿讲 + 会算显存**
- `notes/06_request_lifecycle.md` — 一个请求从出生到结束的完整状态流转
- **集成任务 2**：网关接入 vLLM 的 `/metrics`（Prometheus 格式），把队列深度、TTFT P99 暴露出来——为将来 KEDA 扩缩容埋点

---

## Phase 3：Worker 与模型执行 —— GPU 上真正发生了什么（第 6-7 周）

**目标**：理解一个 batch 如何变成一次 GPU forward。

### 3.1 Worker 与 ModelRunner
- [ ] `vllm/v1/worker/gpu_worker.py`：Worker 的初始化（显存探测、KV cache 形状确定）与执行入口
- [ ] `vllm/v1/worker/gpu_model_runner.py`：**全仓库最核心的大文件**，重点读：
  - `execute_model`：单次 forward 完整流程
  - 输入准备：attention metadata、position ids、`gpu_input_batch.py` 的 batch 组装
  - `cudagraph_dispatcher.py`：CUDA Graph 捕获/重放（为什么 decode 阶段快）
- [ ] `vllm/v1/executor/`：执行器抽象（单卡 UniProcExecutor 为主线）

### 3.2 模型定义与加载
- [ ] `vllm/model_executor/models/qwen3.py` + `llama.py`：挑一个精读——并行 Linear、RMSNorm、RotaryEmbedding 怎么组装
- [ ] `vllm/model_executor/model_loader/`：HF 权重如何加载、分片、映射到并行层
- [ ] `vllm/model_executor/layers/quantization/`：量化入口（配合 14B/32B AWQ 实跑体验）

### 3.3 Attention 后端与采样
- [ ] `vllm/v1/attention/`：attention 后端选择（FlashAttention / FlashInfer 等）
- [ ] `vllm/vllm_flash_attn/`：vLLM 维护的 flash-attn fork
- [ ] `vllm/v1/sample/sampler.py`：温度、top-p/top-k、penalty 的 GPU 实现
- [ ] `vllm/v1/sample/rejection_sampler.py`：投机采样的接受/拒绝逻辑（先有个印象）

**产出物**：
- `notes/07_gpu_model_runner.md` — 一次 forward 的完整旅程
- `notes/08_model_loading.md` — 从 HF 权重到 GPU 上的模型
- `notes/09_attention_backend.md` — attention 后端与采样
- **集成任务 3**：网关 + vLLM 联调 prefix caching——构造共享长前缀的请求（如固定 system prompt），对比开关 `--enable-prefix-caching` 前后的 TTFT，数据写进笔记（学习清单里点名要的简历素材）

---

## Phase 4：高级特性 —— MaaS 场景高频考点（第 8-9 周）

按对「网关 + 本地部署」目标的价值排序：

### 4.1 Structured Output / Tool Use（Agent 场景刚需，优先级最高）
- [ ] `vllm/v1/structured_output/`：JSON Schema / Regex 约束解码
- [ ] `vllm/tool_parsers/`、`vllm/reasoning/`：tool call 解析与推理链
- [ ] 动手：网关暴露一个 function calling demo

### 4.2 Prefix Caching（已在 Phase 3 实操，这里补原理细节）
- [ ] block 哈希、LRU 淘汰、`benchmarks/benchmark_prefix_caching.py`

### 4.3 量化
- [ ] AWQ/GPTQ/FP8 的 trade-off；用 4080 实跑 Qwen3-14B-AWQ，与 8B FP16 对比显存/质量/速度
- [ ] **产出**：笔记《4080 24GB 上的模型选型实验：FP16 vs AWQ》（压测报告第二弹）

### 4.4 Speculative Decoding
- [ ] `vllm/v1/spec_decode/`：draft model / ngram 投机
- [ ] 动手：ngram 投机对 TPOT 的改善实测

### 4.5 LoRA / 多 LoRA
- [ ] `vllm/lora/`、`vllm/v1/worker/lora_model_runner_mixin.py`
- [ ] 场景联想：MaaS「一个底座 + 多个租户 LoRA」的商业模式就对应这个特性

### 4.6 KV Offload 与 PD 分离（了解概念，单卡不实操）
- [ ] `vllm/v1/kv_offload/`、`vllm/v1/simple_kv_offload/`：KV cache 向 CPU/磁盘卸载
- [ ] `examples/disaggregated/`：prefill/decode 分离部署——MaaS 平台架构面试高频
- [ ] `vllm/v1/fault_tolerance/`：容错机制概览

### 4.7 多模态（选做）
- [ ] `vllm/multimodal/`：图像/视频输入处理

**产出物**：
- `notes/10_structured_output.md`、`notes/11_quantization.md`、`notes/12_spec_decode.md`、`notes/13_lora.md`
- **集成任务 4**：网关能力补全——按模型限流（每租户 token bucket）、请求级超时、失败重试到备用实例。至此「网关 + vLLM」已具备 MaaS 平台网关层的基本形态

---

## Phase 5：分布式推理（第 10 周，理论为主）

> 单卡 4080 无法实操 TP/PP，本周以读代码 + 概念输出为主，有条件再租多卡机器验证。

- [ ] `vllm/distributed/parallel_state.py`：TP/PP/DP 通信组管理
- [ ] `vllm/distributed/device_communicators/`：NCCL 封装
- [ ] TP 在模型层的体现：`ColumnParallelLinear` / `RowParallelLinear` 的切分与 all-reduce 位置
- [ ] DP 协调：`vllm/v1/engine/coordinator.py`、`vllm/v1/worker/dp_utils.py`
- [ ] 与 6.824 知识的勾连：TP all-reduce ≈ 同步屏障；DP ≈ 无状态水平扩展——面试时主动建立这个连接

**产出物**：
- `notes/14_distributed.md` — 并行策略速查 + 「什么时候需要 TP/PP/DP」决策树

---

## Phase 6：Kernel 层（第 11 周起，按兴趣深入，非必须）

- [ ] `csrc/attention/`：PagedAttention CUDA kernel（对照论文 kernel 部分）
- [ ] `vllm/compilation/`：`torch.compile` 集成、算子融合、piecewise CUDA Graph
- [ ] Triton kernel：`vllm/kernels/`、`benchmarks/kernels/`（kernel 实验放这里，不放 tests/）
- [ ] 性能分析：`vllm/profiler/` + `nsys`，抓一次 decode step 的 kernel 时间线

**产出物**：
- `notes/15_kernels.md` — kernel 层认知（面试能讲到「融合了什么、为什么省显存带宽」即可）

---

## 贯穿全程：「网关 × vLLM」项目线（最终交付）

每完成一个 Phase，项目就长大一点：

| 里程碑 | 内容 | 依赖 |
|---|---|---|
| M0 | 网关反向代理 + SSE 流式透传 vLLM | Phase 0 |
| M1 | 多模型/多版本路由 | Phase 1 |
| M2 | Prometheus 指标暴露（队列、TTFT、吞吐） | Phase 2 |
| M3 | prefix caching 对比实验数据入库 README | Phase 3 |
| M4 | 租户级限流、超时、重试、function calling 透传 | Phase 4 |
| M5 | README 完工：架构图 + 压测数据 + 设计取舍（当晋升材料写） | 全部 |

**本地部署最终形态**（一台 4080 主机即可演示）：
```
客户端 → Go 网关（鉴权/限流/路由/指标）→ vLLM 实例A（Qwen3-8B，主力模型）
                                      → vLLM 实例B（Qwen3-0.6B，兜底/灰度）
```

---

## 附录 A：主线速查（快速打通版）

```
vllm/entrypoints/cli/serve.py                     # vllm serve 入口
  → vllm/entrypoints/openai/api_server.py          # FastAPI 装配
    → vllm/entrypoints/openai/chat_completion/serving.py   # 请求处理
      → vllm/v1/engine/async_llm.py                # 异步引擎
        → vllm/v1/engine/core.py                   # EngineCore 事件循环（子进程）
          → vllm/v1/core/sched/scheduler.py        # 调度决策
          → vllm/v1/core/kv_cache_manager.py       # KV block 分配
          → vllm/v1/worker/gpu_model_runner.py     # 组装 batch + forward
            → vllm/model_executor/models/qwen3.py  # 模型定义
            → vllm/v1/sample/sampler.py            # 采样
        → vllm/v1/engine/output_processor.py       # detokenize + 返回
```

## 附录 B：关键文件索引

| 模块 | 核心文件 |
|---|---|
| CLI / API 入口 | `entrypoints/cli/serve.py`、`entrypoints/openai/api_server.py`、`entrypoints/openai/chat_completion/` |
| 网关层对照 | `entrypoints/serve/middleware/` |
| Engine | `v1/engine/core.py`、`v1/engine/async_llm.py`、`v1/engine/admission_control.py` |
| 调度器 | `v1/core/sched/scheduler.py` |
| KV Cache | `v1/core/kv_cache_manager.py`、`v1/core/block_pool.py`、`v1/core/kv_cache_utils.py` |
| Worker | `v1/worker/gpu_worker.py`、`v1/worker/gpu_model_runner.py`、`v1/worker/gpu_input_batch.py` |
| 模型 | `model_executor/models/qwen3.py`、`model_executor/model_loader/` |
| 采样 | `v1/sample/sampler.py`、`v1/sample/rejection_sampler.py` |
| 高级特性 | `v1/structured_output/`、`v1/spec_decode/`、`lora/`、`model_executor/layers/quantization/` |
| 分布式 | `distributed/parallel_state.py`、`v1/executor/` |
| 配置 | `vllm/config/`、`vllm/engine/arg_utils.py` |
| 压测 | `benchmarks/benchmark_serving.py`、`benchmarks/benchmark_prefix_caching.py` |

## 附录 C：调试技巧

- **单进程调试**：`VLLM_ENABLE_V1_MULTIPROCESSING=0` 让 EngineCore 不fork子进程，pdb/IDE 断点可直接命中
- **关 CUDA Graph**：`--enforce-eager`，让 forward 逐行可走（性能下降但可调试）
- **详细日志**：`VLLM_LOGGING_LEVEL=DEBUG`
- **小模型快迭代**：学习期一律用 Qwen3-0.6B，启动秒级
- **跟踪一次 forward**：在 `gpu_model_runner.py` 的 `execute_model` 打断点
- **fork 子进程方法**：`VLLM_WORKER_MULTIPROC_METHOD=fork`（默认），调试多进程问题时关注

## 附录 D：知识内化自测清单（每周末过一遍）

- [ ] 能脱稿讲清 PagedAttention：解决什么问题、block 映射、为什么省显存
- [ ] 能脱稿讲清 continuous batching：和 static batching 的差别、iteration-level 调度
- [ ] 能手算：给定模型参数量/层数/hidden size/batch/seq len，KV cache 显存与最大并发
- [ ] 能讲清 TTFT 和 TPOT 分别受什么影响、各自的优化手段（prefix caching / chunked prefill / CUDA Graph / 投机采样）
- [ ] 能讲清调度器抢占：什么时候发生、踢谁、怎么恢复
- [ ] 能画出「网关 → vLLM API 层 → Engine → Worker → Kernel」的完整调用链
- [ ] 能把 Java 高并发经验翻译过来：网关信号量隔离 ↔ vLLM admission control；Sentinel 限流 ↔ MaaS 租户配额

---

## 更新记录

- 2026-05-24：初始版本，基于 V1 架构（V0 尚在，做对比学习）
- 2026-09-29：全面更新——fork 同步至 upstream main；V0 已移除，删除 V0 对比主线；入口层路径更新（`chat_completion/` 子包、`serve/middleware/`）；新增 admission_control、kv_offload、PD 分离等新模块；结合已完成 Go 网关与 4080 24GB 实机环境，新增「网关 × vLLM」贯穿项目线与双机环境策略
