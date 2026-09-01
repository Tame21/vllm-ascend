# v0.25.1 Qwen3.5 LoRA 与 ACL graph 运行时补丁

根据 `patch_mesim.md` 的 TurboManager 机制，将
`patch_qwen3_5_dense_lora.py` 和 `patch_lora_acl_graph.py` 转换成按真实
vLLM / vLLM Ascend 源码层级组织的运行时补丁。原始 `vllm/`、
`vllm-ascend/` 和测试文件均不修改。

## 接入

将 `netrsn_turbo` 目录下的文件按同名路径合入实际工程。复用已有的
`__init__.py`；新建子包按工程打包规则补齐 `__init__.py`，不要覆盖原有
初始化逻辑。

在 TurboManager 的 `version_0251` 分发入口中按以下顺序增加：

```python
import netrsn_turbo.turbo_manager.version_0251.turbo_qwen3_5_dense_lora  # noqa: F401
import netrsn_turbo.turbo_manager.version_0251.turbo_lora_acl_graph  # noqa: F401
import netrsn_turbo.turbo_manager.version_0251.turbo_mamba_postprocess  # noqa: F401
```

Qwen3.5 LoRA 必须先于 ACL graph 补丁加载，因为图隔离复用前者的模型判断、
配置校验和 Punica wrapper 状态。Mamba 补丁与二者独立。

Qwen3.5 LoRA 和 ACL graph 的自动启用条件沿用 v0.23 示例：
`ADAPTATION_PKG_ID` 非空且卡型为 `910B`。已有工程使用其他 LoRA 开关时，
可以在该开关中依次显式调用：

```python
from netrsn_turbo.turbo_manager.version_0251.turbo_qwen3_5_dense_lora import (
    apply_qwen3_5_dense_lora_patch,
)
from netrsn_turbo.turbo_manager.version_0251.turbo_lora_acl_graph import (
    apply_lora_acl_graph_patch,
)

apply_qwen3_5_dense_lora_patch()
apply_lora_acl_graph_patch()
```

显式调用不检查环境变量或卡型，由调用方保证版本、硬件和 LoRA 场景正确。
补丁必须在 LoRA manager、compile wrapper 和 ACL graph 状态初始化之前应用。
spawn 子进程仍需通过工程已有的 `special_init.py` 重新导入。

## Qwen3.5 dense LoRA 映射

| 原补丁目标 | Turbo 实现位置 |
| --- | --- |
| `PunicaWrapperNPU` 的 expand、metadata 与 no-LoRA guard | `turbo/version_0251/vllm_ascend/lora/punica_npu.py` |
| `LoRAModelManager.__init__` 与多模态模块映射 | `turbo/version_0251/vllm/lora/model_manager.py` |
| `WorkerLoRAManager._load_adapter` | `turbo/version_0251/vllm/lora/worker_manager.py` |
| `AscendAttentionBackendImpl.update_graph_params` | `turbo/version_0251/vllm_ascend/attention/attention_v1.py` |

管理入口
`turbo_manager/version_0251/turbo_qwen3_5_dense_lora.py` 在原类上包装方法，
并显式使用 `staticmethod` 保留 attention 方法的描述符。自定义 LoRA expand
算子、分块临时张量、MM key remap、decode `no_lora` 更新和配置限制均保留。
同时回补社区 #11940 的 shrink CopyOut 对齐兼容：旧 AscendC 内核要求 FP32
输出 rank 按 8 个元素（32 bytes）对齐；Python 包装仅对该旧内核将权重和输出
临时 padding 到 8 的倍数，执行后裁回逻辑 rank。PyTorch-native 后端和已对齐
rank 不增加额外操作，fully-sharded TP 的本地 rank 也不会改变 all-gather 语义。

## LoRA ACL graph 映射

| 原补丁目标 | Turbo 实现位置 |
| --- | --- |
| `GraphParams` 与 `weak_ref_workspaces` | `turbo/version_0251/vllm_ascend/compilation/acl_graph.py` |
| `TorchCompileWithNoGuardsWrapper` | `turbo/version_0251/vllm/compilation/wrapper.py` |
| `GPUModelRunner` 的 LoRA dummy run、warmup 与 capture | `turbo/version_0251/vllm/v1/worker/gpu_model_runner.py` |

管理入口 `turbo_manager/version_0251/turbo_lora_acl_graph.py` 使用
`TurboManager.register_patch()` 替换 `GraphParams` 和
`weak_ref_workspaces`，使已导入的同对象引用同步传播。类方法和新增辅助方法
直接安装在原类上。若任一 ACL graph 状态已经初始化则拒绝应用，避免丢失状态。

设置 `VLLM_USE_AOT_COMPILE=1` 时，compile wrapper 会分别编译、加载并原子保存
LoRA、Base 和单 token Base 三份 AOT 产物（`model.lora`、`model.base`、
`model.base_one`）。缓存只命中部分变体时，其余变体会在首次使用时补编译并保存。

## 兼容性边界

不要同时加载 vLLM Ascend 原生的
`patch_qwen3_5_dense_lora.py`、`patch_lora_acl_graph.py` 与本运行时补丁；
两套实现会重复注册同名自定义算子或重复包装方法。该补丁只面向当前 vLLM
0.25.1 / vLLM Ascend 源码组合，其他版本必须重新核对方法签名和图描述符。

当前工作区已完成所有 Python 文件的语法解析、Ruff 检查、补丁目标清单和
Mamba kernel 内容比对。由于没有真实 NetrsnTurbo 与 NPU 环境，部署前仍需
验证 LoRA/Base 交替请求、FULL graph 捕获与重放、MTP、MM LoRA key remap、
无 LoRA decode、图 workspace 弱引用以及 spawn worker 重打补丁。
