# DSpark Decode-Only Prefill 加速实现文档

> 对应设计文档：`dspark_decode_only_prefill_acceleration_plan.md`（仓库外工作区根目录）。
> 本文记录首版落地的实际实现：改动文件、核心数据结构与调用链、关键决策与验证结果。
> 基线：vLLM `v0.25.1`（commit `752a3a5`），vLLM Ascend `releases/v0.25.1rc`（commit `a675940`）。

## 1. 功能概述

将 DSpark 的 context KV 投影与首轮 proposal 从 final prefill 的同步关键路径移出：

```text
prefill_tail（默认，行为不变）:
  final prefill -> DSpark combine/context KV/query/heads -> 返回首 token

decode_only（新）:
  prefill chunk(s)  -> 暂存 raw target 辅助 hidden + positions（无任何 DSpark 计算）
  final prefill     -> 发布 spec_token_ids=[]，立即返回首 token
  第一次普通 decode  -> lazy init（combine + 重建 slot + 写 context KV）-> 首轮 proposal
  steady-state      -> 与现有一致
```

首版仅支持 Model Runner V1（Ascend 默认路径）、同步调度、prefix caching 关闭、无 KV transfer、DCP=1；不支持组合在启动期直接报错，绝不静默回退。

## 2. 改动文件清单

| 文件 | 改动 |
| --- | --- |
| `vllm_ascend/ascend_config.py` | 新增 `DSparkExecutionConfig` 配置类与启动校验；`AscendConfig.__init__` 解析 `additional_config.dspark_config` |
| `vllm_ascend/spec_decode/dspark_proposer.py` | 请求状态机、staging、滑窗裁剪、双上限记账、lazy init、compact/scatter、回退与生命周期管理 |
| `vllm_ascend/ops/triton/spec_decode/utils.py` | 新增 `build_dspark_context_slots_kernel` 批量 slot 重建 kernel 及 Python 包装 |
| `vllm_ascend/worker/model_runner_v1.py` | decode-only 分支、`_update_states` 生命周期 hooks、draft valid mask（含 PP 路径） |
| `vllm_ascend/profiling_config.py` | 登记 `propose_decode_only` profiler 符号 |
| `tests/ut/test_ascend_config.py` | +13 配置校验用例 |
| `tests/ut/spec_decode/test_dspark_proposer.py` | +28 状态机/staging/裁剪/回退/生命周期/compact 用例 |
| `tests/e2e/pull_request/one_card/spec_decode/test_dspark.py` | +`test_dspark_decode_only_matches_prefill_tail` |
| `docs/source/user_guide/feature_guide/speculative_decoding.md` | 新增 "DSpark decode-only prefill" 特性章节 |

未修改：上游 vLLM scheduler / `vllm/v1/request.py` / V2 runner / DSpark 模型结构 / KV connector 协议。

## 3. 配置层（ascend_config.py）

### 3.1 `DSparkExecutionConfig`

```python
DSparkExecutionConfig(dspark_config: dict, vllm_config: VllmConfig)
```

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `execution_phase` | `prefill_tail` | `decode_only` 启用新路径；提供 `decode_only: bool` 便捷属性 |
| `staging_device` | `npu` | 首版仅 `npu` |
| `max_staged_tokens_per_request` | decode-only 必填 | 单请求 retained token 上限 |
| `max_staged_bytes_total` | decode-only 必填 | 单 worker staged 总字节上限 |
| `lazy_init_chunk_tokens` | `4096` | lazy init 单次投影的最大 context token 数 |
| `overflow_policy` | `fallback_prefill_tail` | 首版唯一策略 |

### 3.2 启动期校验（decode_only 时全部强制）

1. `dspark_config` 必须是 dict；
2. `speculative_config.method == "dspark"`；
3. 两个上限与 `lazy_init_chunk_tokens` 为正 int，显式拒绝 `bool`（`isinstance(value, bool)` 先于 int 判断）；
4. `staging_device == "npu"`、`overflow_policy` 合法；
5. `cache_config.enable_prefix_caching` 为 False；
6. `scheduler_config.async_scheduling` 为 False；
7. `kv_transfer_config.is_kv_transfer_instance` 为 False；
8. `vllm_config.use_v2_model_runner` 为 False；
9. `parallel_config.decode_context_parallel_size == 1`。

`prefill_tail` 下不做任何组合校验，保证默认路径零行为变化。

### 3.3 配置解析入口

`AscendConfig.__init__` 中：

```python
dspark_config = additional_config.get("dspark_config", {})
self.dspark_config = DSparkExecutionConfig(dspark_config, vllm_config)
```

proposer 侧优先读单例 `get_ascend_config().dspark_config`；若单例未初始化（极早期构造），直接用 `vllm_config.additional_config` 重新解析一次 `DSparkExecutionConfig`——配置错误仍然会在此处大声失败，而不是被静默降级为 prefill_tail。

## 4. Proposer 核心实现（dspark_proposer.py）

### 4.1 数据结构

```python
class DSparkRequestPhase(Enum):      # COLLECTING / PENDING_INIT / READY /
                                     # FALLBACK_PREFILL / INVALID

@dataclass
class StagedDSparkChunk:             # 一个 prefill chunk 的暂存内容
    raw_hidden_states: torch.Tensor  # combine_hidden_states() 之前的 raw 辅助 hidden
    positions: torch.Tensor          # 与 hidden 行一一对应的 target positions（int32）
    num_tokens / num_bytes / hidden_row_bytes / position_row_bytes

@dataclass
class PendingDSparkContext:          # 每 request 的 decode-only 状态
    request_id, generation           # generation 防 abort 后同 ID 重提复用旧状态
    phase: DSparkRequestPhase
    chunks: list[StagedDSparkChunk]
    num_staged_tokens / num_staged_bytes
    prompt_len, retained_context_tokens, fallback_reason, lazy_init_done

@dataclass
class DSparkDecodeOnlyRequestMeta:   # 每 step 的 CPU 侧请求分类
    request_rows / request_ids / is_prefill / finishes_prefill / prompt_lens
```

关键约束：

- **只暂存 raw hidden 与 positions，不暂存 slot mapping**。滑窗 KV manager 可能已回收旧 block，lazy init 一律用当前 block table 重建。
- 暂存 tensor 一律 `detach().clone()` 拥有独立 storage，不引用下一步会被覆盖的 runner buffer。
- 全局 `total_staged_bytes` 与每请求 `num_staged_bytes` 在新增/裁剪/回退/清理路径成对更新。

### 4.2 请求分类（纯 CPU，无 `.item()` 同步）

```python
meta = classify_decode_only_requests(
    req_ids, num_computed_tokens_cpu, num_prompt_tokens, num_scheduled_tokens)

is_prefill       = num_computed_before < prompt_len
finishes_prefill = is_prefill and num_computed_before + num_scheduled >= prompt_len
```

`_collect_eligible_rows(meta)` 输出三类 eligible 行：

- `PENDING_INIT` 且本 step 是 decode → pending_rows（触发 lazy init）
- `READY` 且 decode → ready_rows
- `FALLBACK_PREFILL` 且 final prefill → fallback_final_rows

`COLLECTING` 的 prefill 行（含 final prefill）一律不参与 proposal；`INVALID` 直接抛错；decode step 遇到 `COLLECTING`/`FALLBACK_PREFILL` 视为状态机不一致，抛错。

### 4.3 prefill staging（`stage_prefill_context`）

对每个 prefill 行：

1. 取或创建 `PendingDSparkContext`（generation 取自 `_request_generations`）；
2. 若请求在 `PENDING_INIT/READY` 又重新进入 prefill（未触发 resume/rewind hook 的隐式 recompute）：释放旧 staging，重置为 `COLLECTING`，计数 `invalidated_recompute`；
3. `FALLBACK_PREFILL` 状态：当前 chunk 直接走 `_project_live_chunk` 增量写 KV，不再暂存；
4. 正常路径：`raw_slice.detach().clone()` + positions 转 int32 clone，追加 chunk 并记账；
5. `_trim_staged_prefix` 滑窗裁剪（见 4.4）；
6. `_staging_limits_exceeded` 检查双上限，超限走 `_fallback_prefill_tail`；
7. final prefill 完成 → `PENDING_INIT`。

**该方法严禁调用** `combine_hidden_states()` / `precompute_and_store_context_kv()` / draft forward / 任何 draft head——由单测 `test_stage_prefill_never_calls_dspark_model` 守护（MagicMock 断言零调用）。

staging 分配失败（RuntimeError/OutOfMemoryError，尚未发生任何 KV 写入）允许按请求回退，计数 `fallbacks_alloc`。

### 4.4 滑窗精确裁剪（`_trim_staged_prefix`）

- `retained_context_tokens` 在 `initialize_attn_backend` 后由所有 draft KV group 的 spec 推导：任一 full-attention layer → `None`（保留全 prompt）；全滑窗 → `max(sliding_window)`；
- 裁剪按 token 粒度：先整 chunk 弹出，再对头部 chunk `raw_hidden_states[overflow:]` / `positions[overflow:]` 切片；
- 字节记账同步扣减 `overflow * (hidden_row_bytes + position_row_bytes)`，不会泄漏。

### 4.5 双上限与回退（`fallback_prefill_tail`）

触发条件：单请求 retained tokens 超上限（`per_request_tokens`）或全局 staged bytes 超上限（`total_bytes`）。

回退按请求执行、不影响同批其他请求：

1. 将触发超限的当前 chunk 也并入 staged 列表；
2. `_project_staged_contexts`（按 `lazy_init_chunk_tokens` 分包）把全部 staged 内容写入当前 DSpark KV；
3. 释放 staging、状态转 `FALLBACK_PREFILL`；
4. 后续 chunk 走 `_project_live_chunk`（同样按 chunk_tokens 分包）只写 context、不跑 proposal；
5. 该请求的 final prefill 参与 proposal（fallback_final_rows），成功后转 `READY`。

### 4.6 lazy init（`initialize_pending_contexts`）

第一次普通 decode 的 target forward 完成后调用：

1. 收集 `PENDING_INIT` 请求及其 full-batch 行号；
2. `_project_staged_contexts`：把多个请求的 chunks 打包为不超过 `lazy_init_chunk_tokens` 的 flat 批（跨请求也可合并，只要单包不超限）；
3. 每包执行 `_project_context_kv`：
   - `model.combine_hidden_states(raw_hidden)`（DSpark 计算首次发生在这里）；
   - 对每个 draft KV group 用 `build_dspark_context_slots`（见 §5）从 **当前** block table 重建 context slots；
   - 按 `_layer_group_idx` 展开为 per-layer slot list；
   - `model.precompute_and_store_context_kv(combined, positions, slots_by_layer)`；
4. 全部包成功后标记 `lazy_init_done`；任何异常：相关请求全部转 `INVALID`、释放 staging、抛出带 request IDs 的 RuntimeError（半初始化不可恢复，禁止自动重试）。

lazy init **不**合并本 decode step 的 target hidden——随后的正常 compact `_propose` 会对本 step accepted hidden 执行 combine + context KV 写入，与 steady-state 逻辑复用，避免重复投影。

### 4.7 compact proposal 与 scatter（`propose_decode_only` / `_propose_compact`）

编排顺序：

```python
stage_prefill_context(prefill rows)          # 有 prefill 行时
initialize_pending_contexts(pending rows)    # 有首个 decode 行时
eligible = pending + ready + fallback_final
if eligible:
    sub_drafts = _propose_compact(eligible)  # 全 eligible 时走 identity 快路径
    scatter 回 full-batch [num_reqs, K] buffer
    valid_mask[row] = True for eligible
    _mark_rows_proposed(...)                 # PENDING_INIT/FALLBACK_PREFILL -> READY（原子切换）
self._last_draft_valid_mask = valid_mask
```

compact 实现（非 identity 时，纯 CPU 索引计算 + GPU `index_select`）：

- 用 `query_start_loc_cpu` 计算 eligible 行的 token 数、新 `query_start_loc`、compact→full 的 token 索引数组；
- 紧凑化 `target_token_ids/positions/hidden/next_token_ids/num_rejected_tokens_gpu/token_indices_to_sample/seq_lens/block_table_tensor/num_computed_tokens_cpu`（`copy.copy(cad)` 后替换字段，不动共享 tensor）；
- **临时替换** proposer 的 `_per_group_block_tables`（按行 `index_select`）与 `_per_group_slot_mappings`（按 compact token `index_select`），供 DSpark first-pass kernel 使用，`finally` 恢复；
- 复用现有 `AscendSpecDecodeBaseProposer._propose` 完成实际 proposal。

返回的 full-batch buffer 中无效行只是占位数值；scheduler 侧由 valid mask 转成真正的 `[]`（见 §6.3）。

### 4.8 生命周期 API

| 方法 | 语义 |
| --- | --- |
| `release_requests(ids)` | finished/cancel：清空 staging、generation+1 |
| `invalidate_requests(ids, reason)` | resume/rewind/recompute：同上并按 reason 计数、warning 日志 |
| `take_last_draft_valid_mask()` | 返回最近一次 proposal 的逐行有效掩码（与 `_draft_token_req_ids` 同生命周期） |

CPU-only 统计（`_dspark_stats`）：pending 请求数、staged tokens/bytes、各类 fallback 与 invalidation 计数——热路径无任何 `torch.npu.synchronize()` / device `.item()`。

## 5. Slot 重建 kernel（ops/triton/spec_decode/utils.py）

```python
@triton.jit
def build_dspark_context_slots_kernel(
    positions_ptr,      # [num_context] 展平的 retained positions
    req_row_map_ptr,    # [num_reqs] compact 行 -> full-batch 行
    req_start_loc_ptr,  # [num_reqs+1] 每请求在展平数组中的 token 起点
    block_table_ptr,    # 当前 full-batch block table（per group）
    block_table_stride,
    out_slots_ptr,      # [num_context] 输出 slots
    block_size, num_reqs,
)
# slot = block_table[full_row, pos // block_size] * block_size + pos % block_size
```

- 每请求一个 program，token 循环内完成 block 查表与 slot 计算，无 Python 循环、无 D2H 同步；
- Python 包装 `build_dspark_context_slots(...)`：`num_reqs == 0` 直接返回；
- 现有 kernel（`copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid` 等）未改动。

`_lazy_init_slot_buffers` 按 group 预分配 `lazy_init_chunk_tokens` 长度；`_project_context_kv` 带显式检查：单包 token 超过 buffer 即抛错（配置错误应启动后尽早暴露，而非越界写）。

## 6. Model Runner V1 改造（model_runner_v1.py）

### 6.1 decode-only 分支

`propose_draft_token_ids()` 的 eagle/draft-model 共享分支内，在 target sampling / rejected 修正 / `prepare_inputs[_padded]` / raw aux hidden 组装（`torch.cat([h[token_indices] for h in aux_hidden_states], dim=-1)`）完成之后、通用 `drafter._propose(...)` 之前：

```python
if self._is_dspark_decode_only():
    return self.drafter.propose_decode_only(...)
```

`_is_dspark_decode_only()` = speculative_config.use_dspark() 且 drafter 是 `AscendDSparkProposer` 且 `drafter.decode_only`。prefill step 因此绝不触发 `combine_hidden_states()`。

### 6.2 生命周期 hooks（`_update_states`）

在 `super()._update_states()` 覆盖 `req_state.num_computed_tokens` **之前**执行：

- `scheduler_output.finished_req_ids` → `drafter.release_requests()`；
- `resumed_req_ids` → `invalidate_requests(reason="resumed")`；
- 非 resumed 但 `num_computed_tokens` 回退（KV-load failure rewind）→ `invalidate_requests(reason="rewind")`；
- 本 step 未调度的请求**不清理**（cached request state 仍可能在后续 step 被调度）。

### 6.3 draft valid mask（含 PP）

- 新增 `self._draft_token_valid_mask_cpu: list[bool] | None`，生命周期与 `_draft_token_req_ids` 一致；
- `_copy_draft_token_ids_to_cpu()` 在快照 req_ids 的同时快照 `drafter.take_last_draft_valid_mask()`（非 decode-only 恒为 None，零开销）；
- `_get_draft_token_ids_cpu()` override：`[row if valid else [] for row, valid in zip(...)]`——final prefill / collecting 行变成真正的空列表，scheduler `update_draft_token_ids` 不会为其生成 speculative lookahead，下一轮该请求只调度 1 个普通 decode token；
- PP 同步路径（`output_spec_token_ids` 直接 `_draft_token_ids.cpu().tolist()` 处）应用同一 mask，避免单卡正确、PP 发布占位 token。

## 7. 每阶段时序（实现后）

```text
非 final prefill chunk:
  target forward -> stage（裁剪/记账）-> valid_mask=False -> publish []
final prefill:
  target sample 首 token -> stage 最后一段 -> COLLECTING→PENDING_INIT -> publish []
第一次普通 decode:
  target 采样第 2 个 token -> lazy init staged prompt context
  -> compact _propose 对本 step hidden 写 context KV + query block
  -> PENDING_INIT→READY -> publish K drafts
steady-state decode:  与 prefill_tail 完全一致
mixed batch:          eligible=[pending/ready/fallback-final] 行 compact 后
                      scatter 回原序，prefill 行保持 []
```

## 8. 测试与验证

### 8.1 UT（CPU 可运行，沙箱实测通过）

`tests/ut/test_ascend_config.py::TestDSparkExecutionConfig`（13 用例）：默认 prefill_tail、decode-only 缺上限/非 dspark method/prefix caching/async/KV transfer/V2/DCP>1 报错、非法 enum、0/负数/bool/float 冒充 int 报错、合法配置通过。

`tests/ut/spec_decode/test_dspark_proposer.py`（+28 用例）：

- staging 零 DSpark model 调用；chunked prefill 按 request_id 累积；tensor 独立 storage；
- final prefill → `PENDING_INIT`；滑窗精确裁剪（positions 断言 `[2,3,4,5]`）；full-attention 不裁剪；
- 单请求 token 超限只回退该请求；全局字节超限计数正确；回退后 chunk 直投；
- `PENDING_INIT/READY` 重入 prefill 重置 `COLLECTING`；
- release/invalidate 清理与 generation 递增；lazy init 失败转 INVALID 并抛错；
- prefill-only step 不调用 `_propose`、mask 全 False；mixed batch compact 的 token/next_token/qsl 断言 + scatter 正确；全 eligible 走 identity 路径。

### 8.2 运行结果

```text
pytest tests/ut/spec_decode/ tests/ut/test_ascend_config.py::TestDSparkExecutionConfig ...
=> 90 passed（沙箱 CPU venv + 仓库 CPU conftest）
ruff check / ruff format => 全部通过
```

a2/（NPU 专用）与少量依赖真实设备的既有用例在沙箱无法收集/运行（`sfa_v1`、设备类型推断等），与本次改动无关。

### 8.3 NPU E2E（待真实环境执行）

`test_dspark_decode_only_matches_prefill_tail`：同一模型分别以 prefill_tail 与 decode_only 启动，断言 greedy 输出 token IDs 逐一致 + acceptance per position 差值 < 0.1。性能验收（TTFT 下降、首 ITL 单列、steady TPOT/acceptance 无回退、峰值显存、fallback 只影响触发请求）按设计文档 §20.5/§21.2 在目标硬件执行。

## 9. 与设计文档的偏差说明

| 设计文档 | 实现 | 原因 |
| --- | --- | --- |
| §8 建议在 base proposer 抽 helper | 全部 orchestration 收敛在 `AscendDSparkProposer` | 避免 base proposer 行为变化；符合"首版不大改共享路径"原则 |
| §11.1 kernel 输入含 compact request-to-full-row mapping | 一致（`req_row_map_ptr`） | — |
| §9.2 snapshot 命名 `_draft_token_valid_mask_cpu` | 一致 | — |
| draft model 能力校验时机（§4.3 要求启动期） | 移至 `initialize_attn_backend`（load_model 之后） | proposer `__init__` 时 `self.model` 尚未加载 |
| 配置校验位置（§4.3 在 AscendConfig 构造期） | 一致；proposer 侧额外兜底重解析 | 防御单例未初始化时 decode-only 被静默丢弃 |

## 10. 模型兼容性：Qwen3 系与 DeepSeek-V4-Flash

两类 DSpark draft 模型的差异点与 decode-only 的处理：

| 差异点 | Qwen3 DSpark | DeepSeek-V4-Flash DSpark | decode-only 处理 |
| --- | --- | --- | --- |
| target positions | xdrope/mrope：`[rope_dim, N]`（如 Qwen3.6 `[3, N]`） | MLA：1-D `[N]` | `_flatten_target_positions()` 取第一 rope 维，**staging、lazy init、`_propose` 全路径统一摊平**——DFlash/DSpark first-pass kernel 按 token flat 索引 positions，不支持 2-D 布局（2-D 会把 rope 坐标交错进 context/query positions，表现为 acceptance ~2% 与设备级故障） |
| context slot 格式 | 1-D `block_id*bs + pos%bs` 直写 KV | 模型内部 `format_dsa_slot_mapping` 把 1-D 转为 `[block_idx, offset]` 后 `dsa_kv_compress_scatter` | lazy-init kernel 统一输出 1-D int32 slot（与现网 prefill_tail 的 first-pass kernel 相同），DSV4 模型侧自行格式化 |
| 滑窗推导 | `SlidingWindowSpec` 或 full attention | `AscendSlidingWindowMLASpec`（继承 `sliding_window: int`） | 统一 `getattr(spec, "sliding_window")`：DSV4 全滑窗 → retained=max(window) 精确裁剪 |
| draft fused buffer | `_num_attn_layers` 等元数据（`_build_fused_kv_buffers`） | 无该属性 | `_ensure_draft_fused_buffers` 对 DSV4 为 no-op；对 Qwen 校验/重建（见 §9 修复记录） |
| `combine_hidden_states` | `fc(aux)` | `main_norm(main_proj(aux))` | 接口一致，直接调用 |
| slot 列表顺序 | sorted(attn_layer_names) → `_layer_group_idx` | 同左；模型按 `layers.values()` 枚举消费 | 与现网 `_context_slot_mapping_buffers` 构造方式完全相同 |

结论：DSV4 无需模型侧适配，decode-only 直接兼容；两类模型均已由 UT 锁定兼容点（mrope 摊平 / DSV4 SW-MLA spec 推导）。仍需在 DSV4 真机执行 §8.3 的 e2e parity 验证。

## 11. 已知限制（首版）

- 仅 Model Runner V1；V2 需按设计文档 §16 独立实现（启动即报错）；
- 仅同步调度、prefix caching 关闭、无 PD/KV transfer、DCP=1；
- 仅 greedy DSpark（沿用 V1 现有限制）；
- TP/DP/PP 保持现有 DSpark 支持水平，需真实硬件多卡正确性测试；
- 性能指标（TTFT/首 ITL）待 NPU 基准；不建议以沙箱结果外推。
