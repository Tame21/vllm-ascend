# LoRA、AOT 与 ACL Graph 隔离补丁

按工作区 `AOT_AND_GRAPH_ISSUES_SUMMARY.md` 移植到当前 vLLM 0.25.1。
补丁由 worker/platform 初始化自动加载，无需安装外置脚本。

## 行为

- `patch_lora_acl_graph.py` 同时处理普通 compile 和 AOT：LoRA、动态 token
  base、单 token base 使用三个 callable。AOT 缓存分别保存为 `model.lora`、
  `model.base`、`model.base_one`，支持部分缓存加载和缺失版本延迟编译。
- FULL ACL Graph 的 events、handles、workspaces、attn_params 按完整
  `BatchDescriptor` 隔离。未捕获的 descriptor 在 replay 阶段报错。
- warmup/capture 的 dummy LoRA 数量跟随 descriptor，base 编译在正式捕获前完成。
- 通用图补丁独立维护 decode 的 `no_lora` 状态。Qwen FIA 更新仅接收具备
  `seq_lens_list` 和 `actual_seq_lengths_q` 的 metadata，不修改原 context。
- `patch_qwen3_5_dense_lora.py` 仅保留 Qwen 语言模型 LoRA 的模块名前缀修复。
  SGMV/BGMV 算子适配、LoRA-A/packed LoRA-B 自定义算子及对齐临时 buffer
  已移除，prefill/decode 使用 `PunicaWrapperNPU` 原有算子实现。
- 平台补丁自动选择 ACL Graph，关闭 `enable_npugraph_ex` 和
  `enable_static_kernel`。模型匹配和配置检查由 `patch/lora_graph_config.py` 共享。

## 适用范围

仅 vLLM 0.25.1、非 310P 的 A2/A3、runner v1、开启 LoRA graph specialization。
Qwen3.5 dense 的 `max_lora_rank` 为 8 或 16；Mistral3-based
Magistral-Small-2509 语言模型 adapter 的 rank 不超过 64。
不支持 context parallelism、DBO/microbatching、sleep mode、非 MTP speculative
decoding；Magistral 视觉塔/connector adapter 不在支持范围内。
无 LoRA、其他模型、eager、关闭 graph 或关闭 LoRA specialization 时保留原路径。

```bash
VLLM_USE_AOT_COMPILE=1 vllm serve /path/to/Qwen3.5-dense \
  --enable-lora --max-lora-rank 16 --lora-modules adapter=/path/to/adapter

VLLM_USE_AOT_COMPILE=1 vllm serve mistralai/Magistral-Small-2509 \
  --enable-lora --max-lora-rank 64 --lora-modules adapter=/path/to/adapter
```

升级后重启全部 worker，首次验证使用新的 `VLLM_CACHE_ROOT`。
不要同时加载旧的外置 LoRA 补丁。未使用到的 AOT variant 可以尚未生成。

## 独立控制三路 AOT

`VLLM_ASCEND_ENABLE_LORA_AOT` 是启动开关，默认 `1`，可设为 `0` 关闭
LoRA/base/base_one 三路 AOT 修复，同时保持 `VLLM_USE_AOT_COMPILE=1`：

```bash
export VLLM_USE_AOT_COMPILE=1
export VLLM_ASCEND_ENABLE_LORA_AOT=0
# 然后使用原来的 vllm serve 启动命令
```

| VLLM_USE_AOT_COMPILE | VLLM_ASCEND_ENABLE_LORA_AOT | 执行方式 |
| --- | --- | --- |
| 0 | 0 或 1 | 普通 compile，保留三路 callable 分发 |
| 1 | 1（默认） | 三路 AOT，缓存为 `model.lora`、`model.base`、`model.base_one` |
| 1 | 0 | 原生单 artifact AOT，编译、加载和保存使用原来的 `model` 缓存 |

此开关不关闭 ACL Graph 参数隔离、decode LoRA 状态维护、Qwen GDN metadata
过滤或 dense LoRA 模块名前缀修复。关闭三路 AOT 后，单 artifact 的 base/adapter 路由问题
不再被修复，交替请求可能产生错误输出，启动日志会提示这一点。
开关在模型初始化时读取；更改后需要重启全部 worker。比较两种模式时建议分别
使用新的 `VLLM_CACHE_ROOT`。

## 验证

在仓库根目录运行 CPU 逻辑回归（需要 torch、pytest）：

```bash
python -m tools.verify_lora_graph_cpu
```

该入口加载真实补丁，以替身隔离 vLLM/NPU 依赖，并运行同一组单元测试。
它验证 Python 分发、缓存、模块名前缀映射及算子入口保持原实现，不能替代
真实 NPU 算子、AOT 序列化加载或 ACL Graph 数值验证。

在完整 Ascend 开发环境运行：

```bash
pytest -q tests/ut/patch/worker/test_patch_lora_acl_graph.py \
  tests/ut/patch/worker/test_patch_lora_aot.py \
  tests/ut/patch/worker/test_patch_qwen3_5_dense_lora.py \
  tests/ut/patch/platform/test_patch_lora_graph_config.py

pytest -sv tests/e2e/pull_request/one_card/lora/test_qwen35_lora_graph.py
```

NPU 用例覆盖普通 compile/AOT、rank 8/16、base→adapter A→base→adapter B、
相同 token 数、单 token decode、混合 batch，以及 AOT 重启强制加载且不重写缓存。
正式部署还需按根目录清单验证实际 TP、MTP、Magistral adapter、与 eager 的数值
一致性及吞吐/显存开销。本地 CPU 测试通过不表示这些硬件测试已完成。
