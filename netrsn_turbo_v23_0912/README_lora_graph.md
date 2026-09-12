# v0.23.0 Qwen3.5/Magistral LoRA 与 ACL Graph 补丁

本目录基于工作区中的 `vllm-23`（tag `v0.23.0`）和
`vllm-ascend-23`（`releases/v0.23.0`，commit `2c59ff104`）源码生成。
它是待合入 `netrsn_turbo` 的运行时补丁，不修改两份上游源码。

## 源码检查结论

0.23 已经具备 `BatchDescriptor.has_lora`、`num_active_loras` 和按 LoRA
数量生成图捕获描述符的能力，也已经按 `max_lora_rank` 创建全零权重槽，加载
真实 LoRA 时只覆盖有效 rank。因而真实 rank 小于配置 rank 时会自动补零，
可以继续使用 Ascend 原生 BGMV/SGMV shrink、expand 和 expand-slice 算子。
本补丁不包含逐 token/逐块矩阵计算的 `custom_op` 替代实现。

但下列问题在这份 0.23 源码中仍然存在：

1. `vllm_ascend.compilation.acl_graph.GraphParams` 仍只按 `num_tokens`
   保存 event、workspace、handle 和 attention 参数。相同 token 数的 Base、
   单 LoRA 和多 LoRA FULL 图会共用可变状态。
2. Ascend `_dummy_run()` 虽然先用 `num_active_loras` 构造 descriptor，进入
   `maybe_dummy_run_with_lora()` 时却又临时强制成 `max_loras`，导致捕获键和
   实际 dummy LoRA 数据不一致。
3. compile wrapper 只有一份无 guard callable，AOT 也只读写一个 `model`
   产物。Base 与 LoRA 路径包含不同的 Python 分支，复用首次 trace 的 callable
   会遗漏 LoRA 计算或让 Base 继续执行 LoRA 算子。
4. decode metadata 不刷新 `PunicaWrapperNPU.no_lora`，该值可能残留自上一次
   prefill；Qwen3.5 的混合 attention metadata 也可能让 FULL 图更新阶段选到
   GDN metadata。
5. Qwen3.5/Mistral3 是多模态外壳，部分只针对文本骨干训练的 PEFT key 缺少
   `language_model` 前缀，需要根据真实模块名做无歧义补齐。

0.23 的 `vllm_ascend.lora.fused_moe` 不反向导入
`vllm_ascend.ops.fused_moe`，后者也不导入 `sync_lora_context`，因此没有
0.25 报错栈中的循环引用。本版本入口不会预先导入 `vllm_ascend.ops`。

## 补丁内容

| 目标 | 文件 |
| --- | --- |
| Base/LoRA/单 token Base 三套 compile 与 AOT 产物 | `turbo/version_v0230/vllm/compilation/wrapper.py` |
| 多模态 LoRA key 映射和 manager 初始化 | `turbo/version_v0230/vllm/lora/model_manager.py`、`worker_manager.py` |
| descriptor 对应的 dummy LoRA 数量和预热 | `turbo/version_v0230/vllm/v1/worker/gpu_model_runner.py` |
| Qwen3.5 attention metadata 过滤 | `turbo/version_v0230/vllm_ascend/attention/attention_v1.py` |
| 按完整 `BatchDescriptor` 隔离 ACL Graph 参数 | `turbo/version_v0230/vllm_ascend/compilation/acl_graph.py` |
| 原生算子 no-LoRA guard、模型识别和边界校验 | `turbo/version_v0230/vllm_ascend/lora/punica_npu.py` |
| TurboManager 注册入口 | `turbo_manager/version_v0230/*.py` |

AOT 缓存文件分别为 `model.lora`、`model.base` 和 `model.base_one`。已有部分
产物时只补编译缺少的变体。ACL Graph 参数使用完整 descriptor 作为物理键，
不再让相同 token 数的 Base/LoRA 图共享 attention 可变状态。

## 接入方式

将本目录的文件按同名路径合入实际 `netrsn_turbo` 包，并在 0.23 版本分发入口
只导入统一入口：

```python
import netrsn_turbo.turbo_manager.version_v0230.turbo_lora_acl_graph  # noqa: F401
```

`turbo_lora_acl_graph` 会先安装 dense LoRA 依赖，再安装图与 AOT 补丁；不要在
同一初始化路径中再次自动导入 `turbo_qwen3_5_dense_lora`。显式调用场景可以用：

```python
from netrsn_turbo.turbo_manager.version_v0230.turbo_lora_acl_graph import (
    apply_lora_acl_graph_patch,
)

apply_lora_acl_graph_patch()
```

自动启用条件为 `ADAPTATION_PKG_ID` 非空且卡型为 `910B`。补丁必须在 LoRA
manager、compile wrapper 和 ACL Graph 参数初始化前安装；spawn worker 仍需
沿用工程现有的重新 import 机制。

## 支持范围与配置

- Qwen3.5 dense：`hf_text_config.model_type == "qwen3_5_text"`。
- Magistral/Mistral3：外层 `model_type == "mistral3"` 且文本配置为
  `model_type == "mistral"`。
- 只支持 model runner v1、910B、非 CP/DBO/微批场景。
- Qwen3.5 只允许 MTP 投机推理；Magistral 的投机推理在本补丁中明确拒绝。
- Ascend 原生 expand 内核只支持配置 rank 8/16/32/64。真实 adapter rank 可以
  更小，但 `max_lora_rank` 必须取上述值并且不小于真实 rank；推荐直接配置成
  能覆盖 adapter 的最小支持值。
- 未验证 sleep mode，当前直接拒绝，避免 sleep/wakeup 重置 descriptor 扩展状态
  时产生未覆盖路径。

## 验证建议

当前机器没有 torch-NPU 运行环境，交付前仍需在目标环境验证：Base/LoRA
交替请求、1 个/多个 adapter、prefill/decode、FULL 图捕获与重放、AOT 首次编译
及缓存重载、Qwen3.5 MTP、Magistral、文本骨干 PEFT key，以及实际 adapter
rank 小于 `max_lora_rank` 时与 eager 基线的 logits 对比。
