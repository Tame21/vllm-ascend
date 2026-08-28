# v0.23 LoRA 运行时补丁

根据 `patch_mesim.md` 的 TurboManager 机制，将 `lora_1.patch` 和
`lora_2.patch` 的非测试修改转换为 Python 运行时补丁。原始 `.patch`、
`vllm/`、`vllm-ascend/` 和测试文件均不修改。

## 接入

当前工作区没有 NetrsnTurbo 的实现。将此目录下的文件按同名路径合入实际
`netrsn_turbo` 包；这是一组补丁模块，不是独立安装包，也不提供替代版
`TurboManager`。复用项目已有的 `__init__.py`，新建的子包按项目打包规则补齐
`__init__.py`，不要覆盖现有初始化逻辑。

在实际 TurboManager 的 `version_0230` 分发入口中，已有 LoRA 补丁加载后增加：

```python
import netrsn_turbo.turbo_manager.version_0230.turbo_lora_graph  # noqa: F401
```

自动启用条件沿用文档示例：`ADAPTATION_PKG_ID` 非空且卡型为 `910B`。
若现有工程使用其他 LoRA 开关，可在该开关内部显式调用：

```python
from netrsn_turbo.turbo_manager.version_0230.turbo_lora_graph import (
    apply_lora_graph_patch,
)

apply_lora_graph_patch()
```

显式调用不检查环境变量/卡型，应由调用方的版本和 LoRA 分支控制。
这些模块针对当前 v0.23 源码；不要仅因版本号更大就直接用于其他版本。

加载顺序保持 `vllm → vllm_ascend → TurboManager`，且必须在任何
`set_*graph*params()` 和图捕获之前完成。重复调用幂等；第一次应用时若已存在
图状态则报错，避免丢弃已捕获的状态。spawn 子进程必须通过工程已有的
`special_init.py` 路径重新导入，不能依赖父进程的 monkey patch。

不要同时叠加原始 `.patch` 和本运行时补丁。若其他本地补丁也覆盖相同方法，
应检查最终加载顺序，避免本补丁再次被覆盖。

## 修改映射

| 原补丁位置 | 运行时实现 |
| --- | --- |
| `lora_1.patch`：`AscendAttentionBackendImpl.update_graph_params` | `turbo/version_0230/vllm_ascend/attention/attention_v1.py` 包装原方法，仅对普通 FIA 的 target 路径过滤无 `seq_lens_list` 的元数据，保留 PA、sinks、draft 分支 |
| `lora_2.patch`：三组 getter/setter/workspace 更新 | `turbo/version_0230/vllm_ascend/compilation/acl_graph.py`，target、draft、draft-prefill 各自分出 Base/LoRA 两套状态 |
| `lora_2.patch`：`ACLGraphWrapper.__call__` 弱引用处理 | 替换 `weak_ref_workspaces`，先从传入的原始状态字典选择当前路由，等价于原补丁在调用点改用 getter |
| `lora_2.patch`：类型及迭代辅助函数 | 复用原 `GraphParams` 类、补齐 workspace 可空注解，向原模块挂载 `GraphParamsByLoRA` 和辅助函数 |
| `lora_2.patch`：sleep/wakeup | `turbo/version_0230/vllm_ascend/device_allocator/sleep_mem_optimized.py` 清理全部六套状态，保留原有 wrapper 缓存清理 |

只有 `CUDAGraphMode.FULL` 且 `batch_descriptor.has_lora` 为真时选择 LoRA；
其他模式、descriptor 缺失或没有 forward context 时选择 Base，与原补丁一致。
事件、workspace、handle 和 attention 参数均独立分配。

所有状态仍保存在原 `vllm_ascend.compilation.acl_graph` 模块中。
已有函数通过 `TurboManager.register_patch()` 注册并统一 `apply_patches()`，
依赖真实 TurboManager 按 `patch_mesim.md` 描述传播到已加载模块中的同对象引用。
新增符号直接挂载，静态方法和类方法显式保留描述符。

FIA 包装仅向原方法传入浅复制后的上下文，原上下文及其 GDN 元数据不变，
后续 conv1d 路径仍可使用。未复制整个 attention 更新或 ACLGraphWrapper 实现。

## 验证边界

已通过 Python 3.10 语法解析；在内存中提取本地上游方法，使用 mock tensor、
NPU 算子和按文档实现的 TurboManager 替身，运行原始补丁中的 6 个用例，
并检查十个函数的注册与已有引用传播、静态/类方法绑定、重复注册、初始化时机、
六套状态隔离、捕获后弱引用、sleep 清理及 FIA 各分支的元数据保留。
验证脚本未作为测试文件加入交付。

本机未安装 torch/NPU，也没有真实 NetrsnTurbo，因此以上是 CPU 隔离验证，
不代表真实补丁引擎或 NPU 捕获/重放已经验证。

交付不含原始 patch 的测试修改。部署前应在实际 NetrsnTurbo + Ascend 环境中
验证真实 TurboManager 的引用传播、Base/LoRA 同 token 数的交替捕获/重放、
GDN/FIA 混合模型、draft/draft-prefill，以及 sleep/wakeup 后重新捕获。
