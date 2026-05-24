# vLLM 源码学习计划（Agent研发方向）

> 目标：逐行吃透 vLLM 推理引擎的核心机制，建立从 API 入口到 GPU Kernel 的完整认知链路。
> 当前版本基于 vLLM V1 新架构（`vllm/v1/`），V0 架构作为对比参考。

---

## 学习原则

1. **先主线后支线**：先打通“请求进来 → 调度 → 推理 → 返回”的完整链路，再深入各模块细节。
2. **代码驱动**：每个阶段必须结合源码文件，用 `gdb` / `pdb` 或加日志的方式跟踪执行流程。
3. **对比学习**：V0 vs V1 架构差异是理解演进的关键。
4. **输出倒逼输入**：每学完一个模块，在 `notes/` 下输出一篇学习笔记。

---

## Phase 1：全局视野与入口层（第 1-2 周）

**目标**：理解 vLLM 作为服务/库的整体形态，掌握 API 入口和核心抽象。

### 1.1 项目结构总览
- [ ] 阅读 `README.md`，了解 PagedAttention 核心思想
- [ ] 梳理目录结构：`vllm/`、`csrc/`、`tests/`、`benchmarks/` 的职责划分
- [ ] 重点区分 V0 架构（`vllm/engine/`、`vllm/core/`）与 V1 架构（`vllm/v1/`）

### 1.2 Python API 入口（离线推理）
- [ ] `vllm/entrypoints/llm.py`：`LLM` 类的完整生命周期
  - `__init__` → `generate` → `encode` 的调用链
  - 与底层 Engine 的交互方式
- [ ] `vllm/sampling_params.py`：采样参数全解析（temperature、top_p、stop 序列等）
- [ ] `vllm/inputs/`：输入预处理流程（prompt tokenization、multi-modal 输入）

### 1.3 OpenAI 兼容 API 入口（在线服务）
- [ ] `vllm/entrypoints/openai/api_server.py`：FastAPI 服务启动流程
- [ ] `vllm/entrypoints/openai/serving_chat.py`：`/v1/chat/completions` 实现
- [ ] `vllm/entrypoints/openai/serving_completion.py`：`/v1/completions` 实现
- [ ] 请求流式（streaming）与非流式的差异处理

**产出物**：
- `notes/01_architecture_overview.md` — 架构总览与模块职责
- `notes/02_api_entrypoints.md` — API 入口层源码解析

---

## Phase 2：V1 Engine 核心调度（第 3-5 周）

**目标**：深入 V1 引擎，理解请求调度、KV Cache 管理、批处理机制。这是 Agent 研发最关心的部分。

### 2.1 Engine 主控流程
- [ ] `vllm/v1/engine/core.py`：`EngineCore` 的核心事件循环
  - `add_request`、`abort_request`、`step` 的实现
  - 与 Scheduler、Worker 的协作关系
- [ ] `vllm/v1/engine/async_llm.py`：异步引擎封装
  - `AsyncLLM` 的 `generate` 异步流实现
  - 与 `EngineCore` 的进程间通信（IPC）
- [ ] `vllm/v1/engine/llm_engine.py`：同步引擎封装

### 2.2 调度器（Scheduler）
- [ ] `vllm/v1/core/sched/`：调度器实现
  - `scheduler.py`：调度策略（连续批处理、抢占、重排）
  - 如何决定每个 step 执行哪些请求
  - `chunked_prefill` 机制：prefill 与 decode 的混合调度
- [ ] 与 V0 调度器（`vllm/core/scheduler.py`）对比

### 2.3 KV Cache 管理（核心中的核心）
- [ ] `vllm/v1/core/kv_cache_manager.py`：KV Cache 分配与回收
- [ ] `vllm/v1/core/block_pool.py`：物理 block 池管理
- [ ] `vllm/v1/core/kv_cache_utils.py`：KV Cache 工具函数（block 映射、前缀缓存）
- [ ] `vllm/v1/core/single_type_kv_cache_manager.py`：单类型 KV Cache 管理
- [ ] **PagedAttention 原理**：逻辑 block → 物理 block 的映射、copy-on-write

### 2.4 请求生命周期
- [ ] `vllm/v1/request.py`：`EngineCoreRequest` 结构
- [ ] `vllm/v1/engine/input_processor.py`：输入预处理（tokenize、build prompt）
- [ ] `vllm/v1/engine/output_processor.py`：输出后处理（detokenize、stop 条件判断）
- [ ] `vllm/v1/engine/detokenizer.py`：增量式 detokenization

**产出物**：
- `notes/03_v1_engine_core.md` — Engine 主控与事件循环
- `notes/04_scheduler.md` — 调度器与连续批处理
- `notes/05_kv_cache.md` — PagedAttention 与 KV Cache 管理
- `notes/06_request_lifecycle.md` — 请求完整生命周期

---

## Phase 3：Worker 与模型执行（第 6-8 周）

**目标**：理解模型在 GPU 上的实际执行流程，从 `model_runner` 到 `attention kernel`。

### 3.1 Worker 架构
- [ ] `vllm/v1/worker/gpu_worker.py`：`GPUWorker` 初始化与执行
- [ ] `vllm/v1/worker/gpu_model_runner.py`：`GPUModelRunner`（~300KB，核心大文件）
  - `execute_model` 方法：单次 forward 的完整流程
  - `prepare_input`：构建 attention metadata、position embeddings
  - ` CUDAGraph` 捕获与重放优化
- [ ] `vllm/v1/worker/worker_base.py`：Worker 抽象基类

### 3.2 Attention 机制
- [ ] `vllm/v1/attention/`：V1 attention 实现
  - `attention.py`：注意力前端接口
  - 与 `vllm/vllm_flash_attn/` 的交互
- [ ] `csrc/attention/`：自定义 CUDA kernel（可选，视 GPU 方向深度）
- [ ] Prefix Caching、FlashAttention、PagedAttention 的协作

### 3.3 模型定义与加载
- [ ] `vllm/model_executor/models/`：主流模型实现（Llama、Qwen、DeepSeek 等）
  - 选择 1-2 个目标模型深入阅读
- [ ] `vllm/model_executor/model_loader/`：权重加载（HuggingFace、GGUF、Sharded）
- [ ] `vllm/model_executor/layers/`：常用层实现（Linear、RMSNorm、RotaryEmbedding）
- [ ] `vllm/model_executor/parameter.py`：参数管理与量化支持

### 3.4 采样与输出
- [ ] `vllm/v1/sample/`：采样实现
  - `sampler.py`：logits 处理、温度缩放、top-p/top-k、repetition penalty
  - `rejection_sampler.py`：speculative decoding 采样

**产出物**：
- `notes/07_gpu_worker.md` — Worker 与 ModelRunner
- `notes/08_attention.md` — Attention 机制与优化
- `notes/09_model_execution.md` — 模型加载与执行流程
- `notes/10_sampling.md` — 采样策略实现

---

## Phase 4：分布式推理（第 9-10 周）

**目标**：理解多卡、多机推理的通信与并行策略。

### 4.1 分布式基础
- [ ] `vllm/distributed/parallel_state.py`：并行状态管理（TP、PP、DP）
- [ ] `vllm/distributed/device_communicators/`：设备通信器（NCCL、MPI）
- [ ] `vllm/distributed/communication_op.py`：通信原语封装

### 4.2 并行策略
- [ ] **Tensor Parallelism (TP)**：`vllm/model_executor/layers/` 中的并行 Linear 层
- [ ] **Pipeline Parallelism (PP)**：`vllm/v1/executor/` 中的 pipeline 执行
- [ ] **Data Parallelism (DP)**：`vllm/v1/worker/dp_utils.py`
- [ ] `vllm/v1/executor/`：V1 执行器实现

**产出物**：
- `notes/11_distributed.md` — 分布式推理原理与实现

---

## Phase 5：高级特性（第 11-13 周）

**目标**：掌握 Agent 场景高频使用的高级功能。

### 5.1 Speculative Decoding（投机采样）
- [ ] `vllm/v1/spec_decode/`：V1 投机解码实现
  - Draft model 推理、验证、接受/拒绝逻辑
  - `vllm/spec_decode/`（V0 实现，对比学习）

### 5.2 Prefix Caching（前缀缓存）
- [ ] `vllm/v1/core/kv_cache_utils.py`：前缀匹配与复用
- [ ] Block 哈希、LRU 淘汰策略

### 5.3 LoRA / Multi-LoRA
- [ ] `vllm/lora/`：LoRA 适配器加载与切换
- [ ] `vllm/v1/worker/lora_model_runner_mixin.py`：V1 LoRA 支持

### 5.4 量化（Quantization）
- [ ] `vllm/model_executor/layers/quantization/`：量化方法（AWQ、GPTQ、FP8、INT8）
- [ ] `csrc/quantization/`：量化 CUDA kernel

### 5.5 Multi-Modal
- [ ] `vllm/multimodal/`：多模态输入处理（图像、视频、音频）
- [ ] Vision Encoder 与 LLM 的交互

### 5.6 Structured Output / Tool Use
- [ ] `vllm/v1/structured_output/`：结构化输出（JSON Schema、Regex 约束）
- [ ] `vllm/tool_parsers/`：Tool Call 解析
- [ ] `vllm/reasoning/`：推理链（Chain-of-Thought）支持

**产出物**：
- `notes/12_speculative_decoding.md`
- `notes/13_prefix_caching.md`
- `notes/14_lora.md`
- `notes/15_quantization.md`
- `notes/16_multimodal.md`
- `notes/17_structured_output.md`

---

## Phase 6：性能优化与 Kernel 层（第 14-16 周）

**目标**：深入底层优化，理解 vLLM 高性能的根本原因。

### 6.1 CUDA Kernel 与 Triton
- [ ] `csrc/`：C++/CUDA 扩展核心
  - `attention/`：PagedAttention CUDA kernel
  - `moe/`：Mixture-of-Experts kernel
  - `quantization/`：量化 kernel
- [ ] `vllm/kernels/` / `vllm/triton_utils/`：Triton kernel
- [ ] `vllm/compilation/`：`torch.compile` 集成与图编译优化

### 6.2 性能分析工具
- [ ] `vllm/profiler/`：内置 profiler
- [ ] `benchmarks/`：官方 benchmark 脚本
- [ ] 使用 `nsys`、`py-spy` 进行端到端性能分析

### 6.3 内存与显存优化
- [ ] `vllm/v1/core/kv_offload/`：KV Cache Offload（CPU/GPU 换入换出）
- [ ] `vllm/device_allocator/`：显存分配器
- [ ] `vllm/sequence.py`：Sequence 状态与内存占用

**产出物**：
- `notes/18_kernels.md` — Kernel 层源码分析
- `notes/19_performance.md` — 性能优化方法论

---

## 附录：推荐学习路径速查

```
快速打通主线（4周精简版）：
  entrypoints/openai/api_server.py
    → v1/engine/async_llm.py
      → v1/engine/core.py
        → v1/core/sched/scheduler.py
        → v1/core/kv_cache_manager.py
          → v1/worker/gpu_model_runner.py
            → model_executor/models/llama.py（选1个模型）
              → v1/sample/sampler.py
                → v1/engine/output_processor.py
```

### 关键源码文件索引

| 模块 | 核心文件 |
|------|---------|
| API 入口 | `entrypoints/openai/api_server.py`、`entrypoints/llm.py` |
| V1 Engine | `v1/engine/core.py`、`v1/engine/async_llm.py` |
| 调度器 | `v1/core/sched/scheduler.py` |
| KV Cache | `v1/core/kv_cache_manager.py`、`v1/core/block_pool.py` |
| Worker | `v1/worker/gpu_worker.py`、`v1/worker/gpu_model_runner.py` |
| Attention | `v1/attention/`、`csrc/attention/` |
| 模型 | `model_executor/models/`、`model_executor/model_loader/` |
| 采样 | `v1/sample/sampler.py` |
| 分布式 | `distributed/parallel_state.py`、`v1/executor/` |
| 配置 | `config/`、`engine/arg_utils.py` |

### 调试技巧
- V1 引擎默认通过子进程运行 `EngineCore`，调试时设置 `VLLM_ENABLE_V1_MULTIPROCESS=0` 可单进程运行
- 使用 `VLLM_LOGGING_LEVEL=DEBUG` 查看详细日志
- 在 `gpu_model_runner.py` 的 `execute_model` 处打断点跟踪单次 forward

---

## 更新记录

- 2026-05-24: 初始版本，基于 V1 架构制定学习路线
