# GDN 混合注意力在投机推理中的 metadata 流转：V1 vs V2 分析与修复

> 背景：model runner v2 开启投机推理时，混合 GDN 模型报
> `GDNAttentionMetadata` 没有 `seq_len` 的 AttributeError。按理
> GDNAttentionMetadata 不应进入 `AscendAutoRegressiveSpeculator._ascend_update_seq_lens`
> 的逐 step seq_lens 记账路径。本文分析 V1 的处理方式、V2 的错误根因、
> 修复方案及精度风险评估。

## 1. 背景：两类 attention metadata 的本质差异

混合 GDN 模型（Qwen3-Next 类）中，GDN 线性注意力层与全注意力层共存，
两种 metadata 的信息模型完全不同：

| | 全注意力（GQA/MLA，`AscendMetadata` 系） | GDN（`GDNAttentionMetadata`，vllm/v1/attention/backends/gdn_attn.py:42） |
|---|---|---|
| KV 长度 | 显式 `seq_lens` / `seq_lens_cpu` / `seq_lens_list`，每 step 需外部推进 | **无任何 seq_lens 字段**，递归状态由 kernel 内部基于 cache indices 推进 |
| 投机推理信息 | 逐 step 修正 seq_lens（FIA kernel 读 KV 长度） | build 期一次性烘焙：`spec_query_start_loc` / `spec_state_indices_tensor` / `num_accepted_tokens` / `spec_sequence_masks` |
| 运行状态 | `attn_state`（Prefill/Decode/SpecDecoding...） | 无 `attn_state` |

关键约束：**speculator 的"逐 step seq_lens +1"记账只对读 KV cache 的
全注意力 kernel 有意义；GDN 不需要也不支持这种记账。**

---

## 2. V1 的处理方式（model_runner_v1.py + llm_base_proposer.py）

V1 的隔离是**结构性**的，靠两层机制：

**(1) GDN 投机信息在 build 期注入，不进 drafter**

model_runner_v1.py:3035 在构建 metadata 时按 builder 类型分发：

```python
if use_spec_decode and isinstance(builder, GDNAttentionMetadataBuilder):
    extra_attn_metadata_args = dict(
        num_accepted_tokens=self.num_accepted_tokens.gpu[:num_reqs_padded],
        num_decode_draft_tokens_cpu=self.num_decode_draft_tokens.cpu[:num_reqs_padded],
    )
```

GDN builder（vllm_ascend/ops/gdn_attn_builder.py:337
`_attach_spec_decode_metadata`）用这些输入一次性生成
`GDNSpecDecodeMetadata`，之后 GDN 对象**不再被修改**。

**(2) drafter 的逐步更新只作用于 CommonAttentionMetadata，然后重新 build**

drafter（`attn_update_stack_num_spec_norm` 等）逐 step 修改的是
**common_attn_metadata**——统一 schema、必然有 `seq_lens` 的中间
对象——每 step 再由各 builder `build()` 生成自己的 typed metadata。
GDN metadata 对象从 build 之后就是只读的，任何 per-step 变更都触碰不到它。

**结论：V1 中"GDN 对象没有 seq_lens"与"逐 step 修改 seq_lens"两条路径
天然不相交，无需任何过滤。**

---

## 3. V2 原来的处理方式与错误原因

V2（worker/v2/ + upstream `AutoRegressiveSpeculator`）为了省去逐 step
重建，**直接在已构建的 per-layer metadata 对象上原地打补丁**。问题在于：
混合模型时 metadata dict（按层名索引）混装两类条目——尤其 **draft prefill
直接复用 target 模型的 attn_metadata**（upstream 设计），以及
`build_draft_attn_metadatas` 从 `model_state.attn_metadata` 按层名过滤——
而 speculator 假设 dict 里全是全注意力 metadata，共 4 处盲写/盲读：

| 位置（修改前） | 操作 | 对 GDN 的后果 |
|---|---|---|
| `_ascend_update_seq_lens` | `attn_meta.seq_lens + 1` | **报错点**：`GDNAttentionMetadata` 无 `seq_lens` → AttributeError |
| `_build_draft_attn_metadata` | 写 `attn_state = DecodeOnly` | dataclass 无 `__slots__`，静默注入垃圾字段 |
| `_init_decode_draft_attn_metadatas` | 写 `attn_state`、`seq_lens_cpu` | 同上，每步污染 |
| `_update_decode_attn_metadata` | 读 `next(iter(...)).seq_lens_cpu.shape` | GDN 层名排序在最前时同样崩溃 |

根因一句话：**V2 把"draft 逐 step 推进 seq_lens"的记账逻辑，错误地应用
到了架构上不需要也不支持该记账的 SSM metadata 上。**

---

## 4. 修复后的 V2

新增 `_iter_full_attn_metadata()` 帮助函数
（vllm_ascend/worker/v2/spec_decode/autoregressive/speculator.py:51）：
遍历时按 `isinstance` 排除 `GDNAttentionMetadata`（及 None），4 处循环
统一改用；`_update_decode_attn_metadata` 的 `next(iter(...))` 改为
`next(_iter_full_attn_metadata(...), None)` 并在无全注意力层时提前返回。

要点：

- **GDN 条目保留在 dict 中，只跳过修改**——draft forward 按层名查
  metadata，混合 draft 模型仍需 GDN 条目随每步浅拷贝；
- GDN 的投机信息供给链路**完全未动**：V2 中
  `MambaHybridModelState.prepare_attn`（vllm/v1/worker/gpu/model_states/mamba_hybrid.py:209）
  通过 `MambaHybridAttnMetadata.get_extra_attn_kwargs` 向 builder 注入
  `num_accepted_tokens` / `num_decode_draft_tokens_cpu`，等价于 V1 的
  runner 注入。

---

## 5. 与 V1 处理逻辑的区别

| | V1 | V2（修复后） |
|---|---|---|
| GDN 与逐 step 记账的隔离方式 | **结构隔离**：per-step 变更只打在 CommonAttentionMetadata（统一 schema）上，再由 builder 重新 build typed metadata | **类型过滤**：统一 metadata dict 混装两类对象，在消费端按 isinstance 排除 GDN |
| GDN 对象生命周期 | build 后只读 | build 后只读（修复恢复了这一不变式，但靠过滤保证而非结构保证） |
| GDN spec 信息注入点 | model runner 构建期（runner buffer） | model state 构建期（`num_accepted_tokens_gpu`，语义一致） |
| 全注意力逐 step 推进 | drafter per-step 重建 | 原地 patch（保留 upstream V2 行为，不变） |

即：修复让 V2 在"原地 patch"的 V2 架构下恢复了 V1 的不变式——
**GDN metadata 生命周期内不被外部修改**；区别仅在保证手段（V1 靠
数据流结构，V2 靠类型白名单过滤）。

### 为什么不用 hasattr 单点跳过

| | 单点 hasattr 方案 | 类型过滤方案（本次修复） |
|---|---|---|
| 覆盖面 | 只修 `_ascend_update_seq_lens` 一处；其余 3 处中 `_update_decode_attn_metadata` 的 `next(iter(...))` 读取在 GDN 层排最前时**仍崩溃**，另两处继续静默污染 GDN 对象 | 4 处统一修复 |
| 语义 | "恰好没这个属性所以跳过"（鸭子类型巧合） | "SSM 层架构上不参与 draft seq_len 记账"（领域规则） |
| 静默吞错 | 跳过**任何**缺属性对象，包括本应有 seq_lens 但构造出错的全注意力 metadata / 未来新 backend——真 bug 被掩盖 | 只放行已知 SSM 类型，其他意外类型快速失败 |

---

## 6. 精度风险评估

**结论：无精度风险，且相对修改前消除了两类隐患。**

1. **GDN 侧数值不变**：被跳过的写操作（`seq_lens`、`attn_state`、
   `seq_lens_cpu`）本来就不是 GDN kernel 读取的字段；GDN 各 draft step
   的行为仍由 build 期烘焙的信息 + kernel 内部状态推进决定，与 V1 语义
   一致。
2. **全注意力侧行为完全不变**：仍然每 step `+1`、仍然写
   `seq_lens_list` / `seq_lens_cpu` / `actual_seq_lengths_q`。
3. **消除了静默污染**：修改前对 GDN 注入的垃圾字段当下虽不被读取，
   但任何未来共享代码 `getattr(metadata, "attn_state")` 做路由会拿到
   假值走错分支——修复后不再产生。
4. **保留快速失败**：过滤是"排除已知 SSM 类型"而非"缺属性就跳过"
   （hasattr 方案）。若未来全注意力 metadata 因 bug 构造不完整，依然
   AttributeError 崩溃，而不是静默跳过导致 FIA 拿旧 KV 长度、draft
   静默出错（那种错误在投机推理中的表现恰是"接受率莫名下降"这类
   难排查的精度事故形态）。
5. **边界覆盖**：纯 GDN batch（无全注意力层）时
   `_update_decode_attn_metadata` 优雅返回；GDN 子类（如 Kimi K3 的
   `KimiK3KDAMetadata`）因继承关系被同样正确排除。

---

## 7. 变更清单

- `vllm_ascend/worker/v2/spec_decode/autoregressive/speculator.py`
  - 新增 `_iter_full_attn_metadata()`；
  - `_ascend_update_seq_lens` / `_build_draft_attn_metadata` /
    `_init_decode_draft_attn_metadatas` / `_update_decode_attn_metadata`
    4 处遍历改用该过滤。
- `tests/ut/worker/test_autoregressive_speculator_gdn.py`（新增回归测试）
  - 覆盖：seq_lens 跳过、decode metadata 更新跳过、纯 GDN 提前返回、
    per-step 拷贝不污染 GDN 条目。
