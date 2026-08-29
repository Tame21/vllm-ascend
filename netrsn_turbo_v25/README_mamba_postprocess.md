# v0.25 Mamba postprocess 运行时补丁

根据 `patch_mesim.md` 的 TurboManager 机制，将 vLLM Ascend 中
`patch_mamba_utils.py` 的单行导入修改转换成 Python 运行时补丁。
原始 `vllm/`、`vllm-ascend/` 和测试文件均不修改。

## 接入

将 `netrsn_turbo` 目录下的文件按同名路径合入实际工程。复用工程已有的
`__init__.py`；新建子包时按工程打包规则补齐 `__init__.py`，不要覆盖现有
初始化逻辑。

在 TurboManager 的 `version_0251` 分发入口中增加：

```python
import netrsn_turbo.turbo_manager.version_0251.turbo_mamba_postprocess  # noqa: F401
```

也可以在 vLLM 和 vLLM Ascend 加载完成后显式调用：

```python
from netrsn_turbo.turbo_manager.version_0251.turbo_mamba_postprocess import (
    apply_mamba_postprocess_patch,
)

apply_mamba_postprocess_patch()
```

补丁模块导入时会自动应用，重复调用幂等。spawn 子进程仍需通过工程已有的
`special_init.py` 重新导入 TurboManager 补丁。

## 修改映射

源文件中的修改只有一行，从原来的：

```python
from vllm_ascend.ops.triton.mamba.postprocess import (
    postprocess_mamba_fused_kernel,
)
```

改成：

```python
from vllm_ascend.patch.worker.patch_mamba_postprocess_v25 import (
    postprocess_mamba_fused_kernel,
)
```

运行时补丁不修改
`vllm_ascend.ops.triton.mamba.postprocess.postprocess_mamba_fused_kernel`
的源码实现；它只将
`vllm_ascend.patch.worker.patch_mamba_utils.postprocess_mamba_fused_kernel`
这个模块级导入引用替换成
`turbo/version_0251/vllm_ascend/ops/triton/mamba/postprocess.py`
中的兼容实现。该目录镜像实际的 vLLM Ascend 功能源码层级，不放在
`vllm_ascend/patch` 安装层中。真实 TurboManager 会按照
`patch_mesim.md` 描述传播同对象引用，因此
`vllm.v1.worker.mamba_utils` 中已经由 `patch_mamba_utils` 赋值的 kernel
引用也会同步替换。这等价于原补丁只改变 import 来源的目的。

不要同时叠加原始源码修改和本运行时补丁。本实现只适用于对应的 vLLM
0.25.x / vLLM Ascend 源码组合，其他版本应重新核对 kernel 签名。

## 验证边界

可在实际 NetrsnTurbo + Ascend 环境中检查补丁应用后两个模块中的
`postprocess_mamba_fused_kernel` 是否为同一对象，并运行 Mamba 推理及推测
解码用例。当前工作区不包含真实 NetrsnTurbo 和 NPU 运行环境，因此本地只做
静态语法、路径和符号检查。
