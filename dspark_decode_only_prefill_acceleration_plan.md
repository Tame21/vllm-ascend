# DSpark Decode-Only Prefill 加速方案

## 1. 背景

当前 DSpark 在 target model 完成 prefill 并采样首个 token 后，会立即执行 draft 上下文 KV 初始化和首轮 proposal。由于 model runner 需要等待 proposer 完成后才返回结果，这部分耗时进入首 token 的同步关键路径，增加 TTFT（Time To First Token）。

当前流程：

```text
Target Prefill
  -> 采样首 token
  -> 初始化 DSpark prompt KV
  -> 执行 DSpark proposal
  -> 返回首 token
```

本方案新增可选的 `decode_only` 模式，把 DSpark 从 prefill 的同步关键路径移到 decode 阶段：

```text
Target Prefill
  -> 采样首 token
  -> 暂存 DSpark 初始化所需上下文
  -> 返回首 token（不生成 draft tokens）

第一次 Decode
  -> 执行一次普通 Target Decode
  -> 延迟初始化 DSpark prompt KV
  -> 执行首轮 DSpark proposal
  -> 返回 token，并为下一轮提供 draft tokens

后续 Decode
  -> Target 并行验证 draft tokens
  -> DSpark proposal
  -> 持续 speculative decoding
```

## 2. 目标与非目标

### 2.1 目标

- prefill 阶段不执行 DSpark 模型计算和 proposal。
- 降低首 token 返回前的同步耗时，改善 TTFT。
- 从第一次 decode 开始启用 DSpark，保持后续 speculative decoding 收益。
- 保持默认行为不变，通过显式配置启用新模式。
- 正确支持请求级状态，避免 mixed prefill/decode batch 相互影响。

### 2.2 非目标

- 不优化 target model 自身的 prefill kernel。
- 不保证降低请求总计算量；DSpark 初始化成本主要从 TTFT 转移到首次 decode。
- MVP 阶段不覆盖 PD 分离、prefix caching 和 async scheduling。

## 3. 收益与代价

预期收益：

- TTFT 减少，下降幅度接近当前 prefill 尾部 DSpark 初始化与 proposal 的耗时。
- prefill 的同步关键路径仅保留 target model forward、采样和必要的上下文暂存。

预期代价：

- 首次 decode 需要执行 DSpark 延迟初始化，首个 TPOT（Time Per Output Token）可能增加。
- 必须临时保存 prompt 对应的 target hidden states、positions 和 slot 信息，会增加短时显存占用。
- 请求状态管理、chunked prefill 和 mixed batch 处理复杂度增加。

因此验收时必须同时观察 TTFT、首个 TPOT、稳态 TPOT、总延迟和显存峰值，不能只看 TTFT。

## 4. 配置设计

使用 `additional_config`，不新增环境变量：

```json
{
  "dspark_config": {
    "execution_phase": "decode_only",
    "context_staging": "gpu",
    "max_staged_tokens": 32768,
    "overflow_policy": "fallback_prefill_tail"
  }
}
```

字段定义：

| 字段 | 可选值 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `execution_phase` | `prefill_tail`、`decode_only` | `prefill_tail` | DSpark 首轮 proposal 的执行阶段 |
| `context_staging` | `gpu` | `gpu` | prefill 上下文暂存位置；MVP 仅支持 GPU |
| `max_staged_tokens` | 正整数 | 待基准测试确定 | 单请求允许暂存的最大 token 数 |
| `overflow_policy` | `fallback_prefill_tail`、`reject` | `fallback_prefill_tail` | 超出暂存上限后的行为 |

配置解析建议放在 `vllm_ascend/ascend_config.py`，增加独立的 `DSparkConfig`，并进行类型和值域校验。默认使用 `prefill_tail`，避免现有用户行为变化。

## 5. 请求状态机

DSpark 是否可执行必须按请求判断，不能只依赖 batch 级 `with_prefill`。

```text
PREFILL_COLLECTING
  | prompt prefill 完成
  v
PENDING_INIT
  | 第一次 decode 完成上下文 KV 初始化
  v
READY
  | 请求完成、取消、抢占或重新 prefill
  v
CLEARED / PREFILL_COLLECTING
```

状态语义：

- `PREFILL_COLLECTING`：请求仍在 prefill，持续收集 chunk hidden states，不执行 DSpark。
- `PENDING_INIT`：prompt 已完成，等待第一次 decode 延迟初始化 DSpark KV。
- `READY`：DSpark KV 已完整，可以正常生成 draft tokens。
- `CLEARED`：请求结束或状态失效，相关暂存数据已经释放。

建议按 `request_id` 维护状态，避免调度重排后依赖 batch index 导致请求错配。

## 6. 上下文暂存设计

严格的 decode-only 模式不能简单跳过 `_propose()`。当前 DSpark 依赖 prompt 的 target hidden states 构造自己的上下文 KV；如果 prefill 阶段既不初始化 KV、也不保存 hidden states，第一次 decode 将缺少完整 prompt 上下文，导致候选 token 错误。

建议增加：

```python
class PendingDSparkContext:
    request_id: str
    hidden_state_chunks: list[torch.Tensor]
    position_chunks: list[torch.Tensor]
    slot_mapping_chunks: list[torch.Tensor]
    num_context_tokens: int
    state: DSparkState
```

实现要求：

- 使用 `query_start_loc` 按请求切分当前 target forward 的 flattened hidden states。
- chunked prefill 每完成一个 chunk，就追加对应 hidden states、positions 和 slot mapping。
- prompt 完成后将状态切换为 `PENDING_INIT`。
- 第一次 decode 初始化完成后立即释放暂存 hidden states。
- 请求完成、取消、抢占、recompute 或 block table 变化时必须清理或重建状态。
- 热路径中避免对 NPU tensor 调用 `.item()`，阶段判断优先复用已有 CPU metadata。

### 6.1 内存控制

额外显存近似为：

```text
prompt_tokens * DSpark hidden_size * dtype_size
```

控制策略：

- 若 DSpark draft attention 已验证为严格滑窗，只保留有效窗口内的 hidden states。
- 未验证为滑窗时，不得擅自截断上下文。
- 设置 `max_staged_tokens`，超出后按照 `overflow_policy` 回退或拒绝请求。
- 后续可增加 pinned CPU staging，但 MVP 不引入额外 H2D 传输路径。

## 7. Proposer 拆分

当前 DSpark proposer 的 `_propose()` 同时承担输入准备、上下文 KV 初始化和候选生成。建议拆分为三个职责：

```python
def stage_prefill_context(...):
    """保存 target 上下文，不执行 DSpark 模型。"""

def initialize_pending_context(...):
    """第一次 decode 时构造并写入 DSpark prompt KV。"""

def propose_decode_tokens(...):
    """仅对 READY 请求执行 DSpark proposal。"""
```

### 7.1 `stage_prefill_context()`

- 只保存请求级上下文。
- 不调用 `precompute_and_store_context_kv()`。
- 不执行 DSpark query block、LM head、Markov head 或 confidence head。
- 对该请求返回空 draft tokens。

### 7.2 `initialize_pending_context()`

- 合并请求在 chunked prefill 中保存的上下文。
- 复用现有 `precompute_and_store_context_kv()` 写入 DSpark draft KV。
- 把第一次普通 decode token 对应的 hidden state 一并纳入上下文。
- 初始化成功后切换为 `READY`，并释放 staged context。
- 初始化失败时清理半成品 KV 和暂存状态，避免后续读取 stale cache。

### 7.3 `propose_decode_tokens()`

- 仅压缩并处理状态为 `READY` 的请求。
- 复用现有 query block、采样和 confidence 逻辑。
- 将结果 scatter 回原 batch；prefill 或 pending 请求保持空 draft。

主要改动文件：

- `vllm_ascend/spec_decode/dspark_proposer.py`
- `vllm_ascend/spec_decode/dflash_proposer.py`（仅在需要抽取通用上下文 KV 接口时修改）
- `vllm_ascend/spec_decode/llm_base_proposer.py`（尽量保持通用路径不受影响）

## 8. Model Runner 调度改造

当前 target forward 完成后，`propose_draft_token_ids()` 会统一调用 drafter。需要在 `vllm_ascend/worker/model_runner_v1.py` 中增加 DSpark decode-only 分支：

```text
对 batch 内每个请求分类：
  prefill              -> stage context，draft=[]
  first normal decode  -> lazy init，然后 proposal
  ready decode         -> 正常 proposal
```

### 8.1 Prefill 请求

- 暂存上下文。
- `_draft_token_ids` 对应行为空或无效。
- scheduler 输出 `spec_token_ids=[]`。
- target model 采样出的首 token 正常返回。

### 8.2 第一次 Decode

- 因为上一轮没有 draft tokens，target model 先执行一次普通 decode。
- 使用 staged prompt context 和本轮 hidden state 初始化 DSpark KV。
- 初始化完成后执行首轮 DSpark proposal。
- proposal 结果在下一调度轮进入 target 并行验证。

### 8.3 Mixed Batch

同一 batch 可能同时包含：

- 正在 prefill 的请求；
- 等待 DSpark 初始化的请求；
- 已经 READY 的 decode 请求。

必须建立请求级 mask，compact 需要执行 proposer 的请求，再把 draft 结果 scatter 回原 batch。不能因为 batch 中存在 prefill 请求就跳过全部 DSpark，也不能让 prefill 请求走 dummy DSpark forward。

## 9. Scheduler 交互

尽量复用现有空 speculative token 语义：

```text
prefill / pending request -> spec_token_ids=[]
ready decode request      -> spec_token_ids=[draft_1, ..., draft_k]
```

预期 scheduler 主体无需改变。实现时需要验证：

- 空 draft 不会被填充为无效 speculative tokens。
- 下一轮会按普通 decode 调度该请求。
- 第一次 decode 后生成的 draft tokens 能正确保存到 request 状态。
- dynamic speculative length 的全局 K 不会覆盖请求级 empty draft。
- 抢占和 recompute 后不会继续使用旧的 DSpark 状态。

如果现有 scheduler 不能稳定表达请求级空 draft，再增加最小的 `dspark_pending_init` 请求标志，不新增独立调度阶段。

## 10. 兼容性边界

### 10.1 MVP 支持范围

- 单机 PD-mix。
- greedy DSpark。
- eager DSpark 路径。
- chunked prefill。
- mixed prefill/decode batch。
- 请求取消、完成和常规抢占清理。

### 10.2 MVP 暂不支持

#### PD 分离

prefill 实例产生的 target hidden states 当前不会随 KV connector 传给 decode 实例。后续需要扩展 connector，传输以下二者之一：

- staged target hidden states；或
- 已经生成的 DSpark context KV。

在完成协议扩展前，`decode_only` 与 PD 分离组合应启动时报错，而不是静默回退。

#### Prefix Caching

命中的 cached prefix 不会重新产生完整 target hidden states，无法直接构造 DSpark prompt KV。MVP 应在启用 decode-only 时关闭 prefix caching 或明确报错。长期方案是为 DSpark context KV 增加可复用的 prefix cache。

#### Async Scheduling

异步调度会增加请求状态、hidden-state 生命周期和 device event 管理复杂度。MVP 先关闭该组合，待同步路径验证完成后再支持。

#### ACLGraph

当前 DSpark proposer 本身使用 eager 路径。decode-only 改造不能扩大这一限制；target model 的 ACLGraph 能力需要单独验证，且不能把服务启动成功等同于首请求成功。

## 11. 异常与回退策略

- staged token 数超过上限：默认回退到当前 `prefill_tail` 路径。
- staged context 分配失败：清理已分配状态并回退，不能保留半初始化 KV。
- 请求抢占或 recompute：清理 DSpark pending/ready 状态，重新收集上下文。
- block table 或 slot mapping 发生不兼容变化：使 pending context 失效并重新初始化。
- 不支持的功能组合：启动时显式报错，避免运行中产生错误候选。

回退必须按请求执行，不能因为单个长请求使整个 batch 退回旧路径。

## 12. 可观测性

增加分阶段统计，便于判断优化是否真正降低 TTFT：

- `target_prefill_ms`
- `dspark_stage_context_ms`
- `dspark_lazy_init_ms`
- `dspark_proposal_ms`
- `first_decode_ms`
- `dspark_staged_tokens`
- `dspark_staged_bytes`
- `dspark_decode_only_fallback_count`

日志只记录状态切换和回退原因，避免逐 token 输出日志。

## 13. 测试方案

### 13.1 单元测试

建议在 `tests/ut/spec_decode/test_dspark_proposer.py` 和 worker 相关测试中覆盖：

1. prefill-only 请求不调用 DSpark model forward。
2. prefill 返回空 draft tokens。
3. chunked prefill 正确累积所有必要上下文。
4. 第一次 decode 只初始化一次 DSpark KV。
5. 后续 decode 不重复初始化。
6. mixed batch 只对 pending/ready decode 请求执行对应操作。
7. proposal 结果正确 scatter 回原请求顺序。
8. 请求完成、取消、抢占和异常后 staged context 被清理。
9. 超过 `max_staged_tokens` 时按策略回退或报错。
10. 默认 `prefill_tail` 行为与修改前一致。

### 13.2 NPU 端到端测试

至少覆盖：

- prompt 长度：1K、8K、32K、128K（硬件允许范围内）。
- batch size：1、8、16。
- chunked prefill 开启和关闭。
- prefill-only、decode-only 和 mixed batch。
- dummy 权重快速门禁与真实权重强制门禁。
- OpenAI-compatible 请求返回 HTTP 200 且输出非空。
- ACLGraph/EP/flashcomm1/MTP 按模型适用性分别记录状态。

真实权重验证必须检查：

- greedy 输出与当前模式一致或满足既定精度标准。
- DSpark acceptance rate 无显著下降。
- 无 draft KV 缺失、slot 越界、位置错误或 stale cache。
- 首请求不是 false-ready。

### 13.3 性能测试

对比 `prefill_tail` 与 `decode_only`：

| 指标 | 目的 |
| --- | --- |
| TTFT | 验证 DSpark 已移出首 token 关键路径 |
| Target prefill latency | 确认 target prefill 本身无回退 |
| 首个 TPOT | 量化延迟初始化的代价 |
| 稳态 TPOT | 确认后续 decode 加速不受影响 |
| E2E latency | 判断成本转移后的总体收益 |
| Throughput | 确认并发吞吐无明显下降 |
| Acceptance rate | 确认 draft 质量不变 |
| Peak NPU memory | 评估 staged context 的显存代价 |

## 14. 验收标准

- `decode_only` 模式下，prefill 阶段没有 DSpark model forward、context KV projection、LM head 或 confidence head 调用。
- TTFT 相比当前模式有可重复的下降。
- 第一次 decode 能正确完成 DSpark KV 初始化，之后进入正常 speculative decoding。
- greedy 输出和 acceptance rate 满足正确性要求。
- mixed batch、chunked prefill、取消和抢占没有状态泄漏。
- staged context 显存受配置上限约束，超限行为可预测。
- 默认配置保持现有行为和性能。
- 真实权重请求返回成功且输出非空；dummy-only 结果不能作为最终通过依据。

## 15. 实施阶段

### 阶段一：MVP

- 增加 `DSparkConfig` 和 `decode_only` 开关。
- 实现请求级状态机与 GPU context staging。
- 拆分 context staging、lazy init 和 proposal。
- 支持同步调度、chunked prefill 和 mixed batch。
- 对 PD 分离、prefix caching 和 async scheduling 显式报错。
- 完成单元测试和真实权重 NPU 验证。

### 阶段二：内存优化

- 根据模型实际 draft attention window 限制 staged context。
- 引入 block 化 staging，避免大量小 tensor 和碎片。
- 评估 pinned CPU staging 与首次 decode H2D 成本。
- 增加动态回退阈值。

### 阶段三：扩展能力

- 为 PD 分离增加 DSpark context 数据传输。
- 支持 prefix cache 复用 DSpark context KV。
- 支持 async scheduling 和跨 step device event 管理。
- 根据性能数据评估首次 decode 初始化与其他计算的并行化。

## 16. 推荐落地顺序

推荐先交付以下受控组合：

```text
decode_only
+ GPU staging
+ 同步调度
+ 非 PD 分离
+ prefix caching 关闭
+ staged-token 上限与自动回退
```

该组合能够直接验证核心假设：将 DSpark 从 prefill 尾部移到第一次 decode，是否能以可接受的首个 TPOT 和显存代价换取稳定的 TTFT 收益。验证通过后，再扩展 PD、prefix caching 和 async scheduling，避免首版同时引入过多状态与传输复杂度。
