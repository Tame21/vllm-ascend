# Ascend 量化开发学习与实践清单

> 基于当前工作区 `vllm` 与 `vllm-ascend` 源码整理。
>
> 使用方式：按推荐顺序逐项完成。完成一个任务后将 `[ ]` 改成 `[x]`，并在对应的“实践记录”中填写代码、实验数据、结论或问题链接。

## 学习目标

完成本清单后，应具备以下能力：

- 判断模型和业务适合哪种量化方案。
- 生成、检查和运行 Ascend 量化权重。
- 沿配置、权重加载、量化方法和 NPU 算子链路定位问题。
- 为新模型适配已有量化算法。
- 为 Linear、MoE 或 Attention 接入新量化算法。
- 完成精度、性能、显存和兼容性验收。

## 总体进度

| 阶段 | 内容 | 状态 | 完成日期 |
| --- | --- | --- | --- |
| 1 | 量化数学与数值基础 | 未开始 | |
| 2 | Ascend 软件栈与性能基础 | 未开始 | |
| 3 | vLLM Ascend 量化主链路 | 未开始 | |
| 4 | W8A8 Dynamic 入门实战 | 未开始 | |
| 5 | 静态、INT4 与 MXFP | 未开始 | |
| 6 | MoE 与 KV Cache 量化 | 未开始 | |
| 7 | 图融合与自定义算子 | 未开始 | |
| 8 | 精度与性能验收 | 未开始 | |
| 9 | 新模型/新算法适配 | 未开始 | |
| 10 | 毕业项目 | 未开始 | |

---

## 第一阶段：量化数学与数值基础

### 1.1 数值表示

- [ ] 理解 FP32、FP16、BF16、INT8、INT4 的位宽、动态范围和精度。
- [ ] 理解符号位、指数位、尾数位对浮点数表示范围的影响。
- [ ] 理解 FP8 E4M3、MXFP8、MXFP4 与普通整数线性量化的差异。
- [ ] 理解溢出、下溢、舍入和饱和截断。
- [ ] 能估算模型权重、激活和 KV Cache 的理论显存占用。
- [ ] 能区分权重存储压缩和真实低比特计算。
- [ ] 能解释低比特模型为什么不一定更快。

必须掌握的基本公式：

```text
# 对称量化
q = clamp(round(x / scale), qmin, qmax)
x_hat = q * scale
scale = max(abs(x)) / qmax

# 非对称量化
q = clamp(round(x / scale) + zero_point, qmin, qmax)
x_hat = (q - zero_point) * scale

# 量化矩阵乘近似关系
Y ≈ (Xq - Zx) @ (Wq - Zw) * Sx * Sw
```

实践任务：

- [ ] 使用 PyTorch 手写一个对称 INT8 量化与反量化函数。
- [ ] 使用 PyTorch 手写一个非对称 INT8 量化与反量化函数。
- [ ] 对随机 Tensor 计算 MSE、最大绝对误差、相对误差和余弦相似度。
- [ ] 人为加入异常值，观察 scale 和量化误差变化。
- [ ] 分别使用 FP16、BF16、INT8 保存同一个 Tensor，比较存储大小。

阶段验收：

- [ ] 给定任意浮点 Tensor，能解释如何选择 scale 和 zero point。
- [ ] 能独立计算一个量化值及其反量化结果。
- [ ] 能解释异常值为什么会降低量化精度。

实践记录：

```text
代码位置：
实验输入：
实验结果：
主要结论：
遗留问题：
```

### 1.2 量化粒度

- [ ] 理解 Per-Tensor：整个 Tensor 共用一个 scale。
- [ ] 理解 Per-Token：每个 token 或每一行使用一个激活 scale。
- [ ] 理解 Per-Channel：每个输出通道使用一个权重 scale。
- [ ] 理解 Per-Group：一组权重元素共用一个 scale。
- [ ] 理解 Per-Block/Microscaling 和 MXFP block scale。
- [ ] 理解粒度越细通常精度越高，但 metadata、访存和算子实现越复杂。
- [ ] 能根据 Tensor shape 写出不同粒度 scale 的预期 shape。
- [ ] 理解 TP 切分可能如何影响 scale 和 group 边界。

常见方案：

| 方案 | 权重 | 激活 | 常见粒度 |
| --- | --- | --- | --- |
| W8A16 | INT8 | FP16/BF16 | 权重 Per-Channel |
| W8A8 | INT8 | INT8 | 权重 Per-Channel，激活 Per-Tensor |
| W8A8_DYNAMIC | INT8 | INT8 | 权重 Per-Channel，激活 Per-Token |
| W4A16 | INT4 | FP16/BF16 | 权重 Per-Group |
| W4A8_DYNAMIC | INT4 | INT8 | 权重 Per-Group，激活 Per-Token |
| W4A4 | INT4 | INT4 | 通常需要平滑、旋转或 FlatQuant |
| MXFP8/MXFP4 | 低比特浮点 | 低比特浮点 | Block/Microscaling |

实践任务：

- [ ] 对同一 Tensor 实现 Per-Tensor 和 Per-Channel 量化。
- [ ] 对激活矩阵实现 Per-Token 动态量化。
- [ ] 比较不同粒度的误差、scale 数量和额外存储。
- [ ] 给定 `[tokens, hidden_size]` 激活，写出 Per-Token scale shape。
- [ ] 给定 `[out_features, in_features]` 权重，写出 Per-Channel scale shape。

### 1.3 静态、动态和混合量化

- [ ] 理解静态量化在校准阶段确定激活 scale。
- [ ] 理解动态量化在推理阶段根据当前输入计算 scale。
- [ ] 理解 Weight-only 量化。
- [ ] 理解混合精度和敏感层回退。
- [ ] 理解静态量化的性能优势与校准依赖。
- [ ] 理解动态量化的精度优势与运行时成本。
- [ ] 理解 Prefill 和 Decode 可能适合不同量化策略。
- [ ] 阅读 [`w8a8_pdmix.py`](vllm-ascend/vllm_ascend/quantization/methods/w8a8_pdmix.py)。

阶段验收：

- [ ] 能根据业务输入分布、精度目标和延迟目标初步选择静态或动态量化。
- [ ] 能解释 PD 分离场景为什么可能采用不同的 Prefill/Decode 量化策略。

### 1.4 PTQ、校准和误差分析

- [ ] 理解 PTQ、QAT、GPTQ、AWQ、SmoothQuant、FlatQuant 和 QuaRot 的主要目标。
- [ ] 理解推理框架主要消费量化权重，而离线量化通常由外部工具完成。
- [ ] 会选择有代表性的校准数据。
- [ ] 理解校准样本数量、序列长度、语言和业务领域的影响。
- [ ] 会分析权重异常值、激活异常值和通道敏感性。
- [ ] 会制定敏感层回退策略。
- [ ] 理解 `lm_head`、MoE gate 和部分 down projection 可能对量化敏感。

源码阅读：

- [ ] 阅读 [`w8a8_int8.py`](vllm-ascend/examples/quantization/llm-compressor/w8a8_int8.py)。
- [ ] 阅读 [`w8a8_int8_dynamic.py`](vllm-ascend/examples/quantization/llm-compressor/w8a8_int8_dynamic.py)。
- [ ] 阅读 [`w8a8_int8_dynamic_moe.py`](vllm-ascend/examples/quantization/llm-compressor/w8a8_int8_dynamic_moe.py)。
- [ ] 阅读 [`w4a8_dynamic_moe.py`](vllm-ascend/examples/quantization/llm-compressor/w4a8_dynamic_moe.py)。

---

## 第二阶段：Ascend 软件栈与性能基础

### 2.1 Ascend 软件栈

- [ ] 理解 Ascend NPU、驱动、固件和 CANN 的关系。
- [ ] 理解 PyTorch → `torch_npu` → ACLNN/CANN 算子的调用路径。
- [ ] 能查看 NPU 型号和健康状态。
- [ ] 能确认驱动、固件、CANN、`torch_npu`、vLLM 和 vLLM Ascend 版本。
- [ ] 理解 A2、A3/A5、310P 的能力可能不同。
- [ ] 理解部分 FP8/MXFP dtype 依赖设备和 CANN 版本。
- [ ] 能区分算法未支持、算子未支持和软件版本不兼容。
- [ ] 阅读 [`vllm-ascend/AGENTS.md`](vllm-ascend/AGENTS.md) 的 NPU 开发要求。
- [ ] 浏览 [`_310p/quantization`](vllm-ascend/vllm_ascend/_310p/quantization) 独立适配。

环境记录：

```text
NPU 型号：
卡数：
驱动版本：
固件版本：
CANN 版本：
Python 版本：
PyTorch 版本：
torch_npu 版本：
vLLM 版本：
vLLM Ascend 版本：
```

### 2.2 NPU 性能基本功

- [ ] 理解 CPU↔NPU 数据传输成本。
- [ ] 理解在 NPU Tensor 上调用 `.item()` 会触发同步。
- [ ] 避免在热路径中频繁进行设备同步。
- [ ] 理解算子启动、动态图和 ACL Graph 的成本。
- [ ] 理解量化、反量化、重排和 scale 读取也会消耗时间。
- [ ] 能区分计算瓶颈与内存带宽瓶颈。
- [ ] 会观察峰值显存和内存碎片。
- [ ] 会测量 TTFT、TPOT、吞吐和 Prefill/Decode 性能。
- [ ] 使用真实输入长度和并发进行测试。

实践任务：

- [ ] 编写一个包含 `.item()` 的 NPU 小程序并测量同步开销。
- [ ] 对比逐元素同步和批量同步。
- [ ] 对比 eager 与图模式下相同工作负载的耗时。
- [ ] 建立统一的性能测试记录模板。

---

## 第三阶段：vLLM Ascend 量化主链路

完整调用链：

```text
模型配置/命令行
  → 创建 QuantizationConfig
  → 读取量化描述
  → 将模型层名映射到量化权重名
  → 根据 quant_type + layer_type 选择 Scheme
  → create_weights
  → load_weights
  → process_weights_after_loading
  → apply
  → torch_npu / ACLNN / 自定义 Ascend C 算子
```

### 3.1 vLLM 与 vLLM Ascend 的边界

- [ ] 理解 vLLM 提供的通用模型层、量化配置接口和权重加载流程。
- [ ] 理解 vLLM Ascend 通过硬件插件注册 Ascend 量化实现。
- [ ] 理解 `--quantization ascend` 的作用。
- [ ] 理解 `QuantizationConfig`。
- [ ] 理解 `QuantizeMethodBase`。
- [ ] 理解 Linear、MoE 和 Attention 方法的接口差异。
- [ ] 理解设备适配为什么不应直接侵入上游模型文件。

必读文档：

- [ ] 阅读 [`量化适配设计文档`](vllm-ascend/docs/source/developer_guide/Design_Documents/quantization.md)。
- [ ] 阅读 [`量化使用指南`](vllm-ascend/docs/source/user_guide/feature_guide/quantization.md)。
- [ ] 能根据设计文档画出量化方法选择流程。

### 3.2 ModelSlim 配置加载

ModelSlim 量化模型的关键描述文件：

```text
quant_model_description.json
```

- [ ] 理解每个权重 key 如何标记 `FLOAT`、`W8A8_DYNAMIC` 等类型。
- [ ] 理解 weight、scale、offset 和 packed weight 的命名约定。
- [ ] 理解普通浮点权重不能被错误地当作量化权重加载。
- [ ] 理解模型配置名称与 vLLM 运行时名称之间的映射。
- [ ] 理解 `hf_to_vllm_mapper` 如何修改量化配置 key。
- [ ] 理解 fused QKV、gate/up 和 MoE experts 的 packed mapping。
- [ ] 理解 fused module 各 shard 的量化类型一致性要求。
- [ ] 理解层级 `FLOAT` 回退。
- [ ] 理解 KV Cache、FA 和 indexer 的量化 metadata。

核心源码：

- [ ] 阅读 [`modelslim_config.py`](vllm-ascend/vllm_ascend/quantization/modelslim_config.py)。
- [ ] 阅读 [`quant_parser.py`](vllm-ascend/vllm_ascend/quantization/quant_parser.py)。
- [ ] 阅读 [`quant_type.py`](vllm-ascend/vllm_ascend/quantization/quant_type.py)。
- [ ] 阅读 [`test_modelslim_config.py`](vllm-ascend/tests/ut/quantization/test_modelslim_config.py)。
- [ ] 阅读 [`test_quant_parser.py`](vllm-ascend/tests/ut/quantization/test_quant_parser.py)。

重点函数阅读顺序：

- [ ] `AscendModelSlimConfig.load_quant_config`
- [ ] `AscendModelSlimConfig.apply_vllm_mapper`
- [ ] `AscendModelSlimConfig.quant_prefix_mapper`
- [ ] `get_linear_quant_type`
- [ ] `get_quant_type_for_layer`
- [ ] `create_scheme_for_layer`
- [ ] `AscendModelSlimConfig.get_quant_method`
- [ ] `AscendModelSlimConfig._add_kvcache_quant_metadata`

阶段验收：

- [ ] 给定一个 Linear 层 prefix，能手工推导最终选中的 Scheme。
- [ ] 能解释错误配置 key 为什么导致无法确定量化类型。
- [ ] 能解释 fused QKV 某个 shard 缺失或类型不一致为何会失败。
- [ ] 能为一个新模型判断是否需要增加 prefix 或 packed module mapping。

调用链记录：

```text
测试模型：
层 prefix：
量化描述 key：
映射后 key：
quant_type：
layer_type：
最终 Scheme：
最终 NPU 算子：
```

### 3.3 Scheme 注册与 Adapter

- [ ] 理解注册表以 `(quant_type, layer_type)` 为 key。
- [ ] 理解同一量化算法可以分别实现 Linear、MoE 和 Attention。
- [ ] 理解未注册算法如何产生 `NotImplementedError`。
- [ ] 理解 Adapter 如何把 Scheme 接入 vLLM 接口。
- [ ] 理解配置、算法和运行层类型必须一致。

核心源码：

- [ ] 阅读 [`registry.py`](vllm-ascend/vllm_ascend/quantization/methods/registry.py)。
- [ ] 阅读 [`base.py`](vllm-ascend/vllm_ascend/quantization/methods/base.py)。
- [ ] 阅读 [`method_adapters.py`](vllm-ascend/vllm_ascend/quantization/method_adapters.py)。
- [ ] 阅读 [`test_registry.py`](vllm-ascend/tests/ut/quantization/methods/test_registry.py)。
- [ ] 阅读 [`test_method_adapters.py`](vllm-ascend/tests/ut/quantization/test_method_adapters.py)。

实践任务：

- [ ] 写一个最小测试 Scheme 并注册。
- [ ] 验证重复注册会报错。
- [ ] 验证未知 `(quant_type, layer_type)` 返回未注册结果。
- [ ] 从 `get_quant_method` 跟踪到具体 Adapter。

### 3.4 量化方法生命周期

#### `create_weights`

- [ ] 理解量化权重、scale、offset 和 bias 的注册。
- [ ] 理解 Parameter 的 shape、dtype 和附加属性。
- [ ] 理解 TP 切分后参数 shape。
- [ ] 理解 packed INT4 的物理 shape 与逻辑 shape。
- [ ] 理解 MoE expert 维度。

#### `process_weights_after_loading`

- [ ] 理解权重转置。
- [ ] 理解 INT4 packing/unpacking。
- [ ] 理解 dtype 转换。
- [ ] 理解设备专用格式转换，例如 NZ。
- [ ] 理解 scale/offset 整理。
- [ ] 理解 fused projection 重排。
- [ ] 理解为何加载后变换不能放到每次 forward 中执行。

#### `apply`

- [ ] 理解静态和动态激活量化。
- [ ] 理解量化 MatMul/Grouped MatMul 输入契约。
- [ ] 理解 bias 处理。
- [ ] 理解反量化和输出 scale。
- [ ] 理解 TP collective。
- [ ] 理解输出 shape 和 dtype 恢复。

建议阅读顺序：

- [ ] [`w8a16.py`](vllm-ascend/vllm_ascend/quantization/methods/w8a16.py)
- [ ] [`w8a8_dynamic.py`](vllm-ascend/vllm_ascend/quantization/methods/w8a8_dynamic.py)
- [ ] [`w8a8_static.py`](vllm-ascend/vllm_ascend/quantization/methods/w8a8_static.py)
- [ ] [`w4a16.py`](vllm-ascend/vllm_ascend/quantization/methods/w4a16.py)
- [ ] [`w4a8.py`](vllm-ascend/vllm_ascend/quantization/methods/w4a8.py)

---

## 第四阶段：W8A8 Dynamic 入门实战

这是推荐首先完整吃透的量化方案。

### 4.1 原理

- [ ] 理解权重 INT8 Per-Channel。
- [ ] 理解激活 INT8 Per-Token。
- [ ] 理解 forward 时动态计算 activation scale。
- [ ] 理解 scale shape 如何随 token 数变化。
- [ ] 理解动态量化与量化 MatMul 的衔接。
- [ ] 分别掌握 Linear 和 MoE 路径。

### 4.2 源码与测试

- [ ] 逐行阅读 [`w8a8_dynamic.py`](vllm-ascend/vllm_ascend/quantization/methods/w8a8_dynamic.py)。
- [ ] 阅读 [`A2 W8A8 Dynamic 测试`](vllm-ascend/tests/ut/quantization/methods/a2/test_w8a8_dynamic.py)。
- [ ] 阅读 [`Qwen3 W8A8 E2E`](vllm-ascend/tests/e2e/pull_request/one_card/test_qwen3_8b_w8a8.py)。
- [ ] 找到 Linear Scheme 的 `create_weights`。
- [ ] 找到 Linear Scheme 的 `process_weights_after_loading`。
- [ ] 找到 Linear Scheme 的 `apply`。
- [ ] 找到动态量化 NPU 算子。
- [ ] 找到量化 MatMul NPU 算子。
- [ ] 找到 MoE Scheme 与 Grouped MatMul 路径。

### 4.3 实验

- [ ] 准备 BF16 基线模型。
- [ ] 准备对应 W8A8 Dynamic 模型。
- [ ] 检查量化描述文件。
- [ ] 完成离线推理。
- [ ] 完成在线服务推理。
- [ ] 固定 prompt 和采样参数比较生成结果。
- [ ] 比较 logits 或中间层余弦相似度。
- [ ] 比较权重显存。
- [ ] 比较 TTFT、TPOT 和吞吐。
- [ ] 分析动态量化算子开销。

阶段验收：

- [ ] 给定任意 W8A8 Dynamic Linear 层，能描述从配置到 NPU MatMul 的完整调用链。
- [ ] 能说明权重、激活和 scale 的 dtype 与 shape。
- [ ] 能定位权重加载错误、scale shape 错误和算子 dtype 错误。
- [ ] 能独立运行相关 UT 和至少一个真实模型 E2E。

实验记录：

```text
模型：
设备：
量化工具：
量化方案：
数据集：
BF16 精度：
W8A8 Dynamic 精度：
BF16 显存：
W8A8 Dynamic 显存：
BF16 TTFT/TPOT/吞吐：
W8A8 Dynamic TTFT/TPOT/吞吐：
结论：
```

---

## 第五阶段：静态、INT4 与 MXFP

### 5.1 W8A8 Static

- [ ] 理解校准生成 activation scale/offset。
- [ ] 理解对称和非对称静态激活量化。
- [ ] 理解 activation scale 与 weight scale 的加载。
- [ ] 检查 scale 广播方向。
- [ ] 理解 Prefill/Decode 分布变化带来的风险。
- [ ] 对比静态与动态方案的精度和性能。
- [ ] 阅读 [`w8a8_static.py`](vllm-ascend/vllm_ascend/quantization/methods/w8a8_static.py)。
- [ ] 阅读 [`A2 W8A8 Static 测试`](vllm-ascend/tests/ut/quantization/methods/a2/test_w8a8_static.py)。

### 5.2 INT4 与权重打包

- [ ] 理解两个 INT4 如何装入一个 INT8 或更大整数容器。
- [ ] 理解 signed INT4 符号扩展。
- [ ] 理解 pack factor。
- [ ] 区分逻辑维度和物理维度。
- [ ] 理解 group size 对精度和 scale 数量的影响。
- [ ] 理解 W4A16、W4A8 和 W4A4 的计算链差异。
- [ ] 检查 TP 切分后 group 边界。
- [ ] 理解 MoE expert 权重打包。
- [ ] 理解 W4A4 对 FlatQuant、旋转或高精度回退的依赖。

源码阅读：

- [ ] [`w4a16.py`](vllm-ascend/vllm_ascend/quantization/methods/w4a16.py)
- [ ] [`w4a8.py`](vllm-ascend/vllm_ascend/quantization/methods/w4a8.py)
- [ ] [`w4a4_flatquant.py`](vllm-ascend/vllm_ascend/quantization/methods/w4a4_flatquant.py)
- [ ] [`w4a4_laos_dynamic.py`](vllm-ascend/vllm_ascend/quantization/methods/w4a4_laos_dynamic.py)
- [ ] [`w4a4_mxfp4.py`](vllm-ascend/vllm_ascend/quantization/methods/w4a4_mxfp4.py)
- [ ] [`w4a16_mxfp4.py`](vllm-ascend/vllm_ascend/quantization/methods/w4a16_mxfp4.py)

实践任务：

- [ ] 手写 INT4 pack/unpack。
- [ ] 验证 pack/unpack 前后逻辑值一致。
- [ ] 对比不同 group size 的误差和 scale 存储。
- [ ] 画出 W4A8 的激活量化、MatMul 和反量化流程。

### 5.3 FP8、MXFP8 与 MXFP4

- [ ] 理解 FP8 和 INT8 的编码差异。
- [ ] 理解 E4M3 的指数/尾数权衡。
- [ ] 理解 MX scale dtype 和 block scale。
- [ ] 能确认目标设备是否支持对应 dtype。
- [ ] 理解 round/rint 等舍入模式。
- [ ] 理解 rollback quant type。
- [ ] 检查 scale 布局和 block 对齐。
- [ ] 理解动态 MX quant 与 MatMul 的衔接。

源码阅读：

- [ ] [`fp8.py`](vllm-ascend/vllm_ascend/quantization/methods/fp8.py)
- [ ] [`fp8_config.py`](vllm-ascend/vllm_ascend/quantization/fp8_config.py)
- [ ] [`w8a8_mxfp8.py`](vllm-ascend/vllm_ascend/quantization/methods/w8a8_mxfp8.py)
- [ ] [`w8a8fp8_dynamic.py`](vllm-ascend/vllm_ascend/quantization/methods/w8a8fp8_dynamic.py)
- [ ] [`quant_parser.py`](vllm-ascend/vllm_ascend/quantization/quant_parser.py)
- [ ] [`test_w8a8_mxfp8.py`](vllm-ascend/tests/ut/quantization/methods/test_w8a8_mxfp8.py)
- [ ] [`test_w8a8fp8_dynamic.py`](vllm-ascend/tests/ut/quantization/methods/test_w8a8fp8_dynamic.py)

---

## 第六阶段：MoE 与 KV Cache 量化

### 6.1 MoE 量化

- [ ] 理解 token routing 和 expert id。
- [ ] 理解每个 expert 的 token 数。
- [ ] 区分普通 MatMul 和 Grouped MatMul。
- [ ] 理解专家权重 `[E, ...]` 的存储。
- [ ] 理解 TP/EP 对专家权重的切分。
- [ ] 理解 gate 的量化敏感性。
- [ ] 理解 gate/up/down projection 的融合和 packed mapping。
- [ ] 理解逻辑 expert id 与本地 expert id。
- [ ] 处理空 expert。
- [ ] 处理极不均匀 token routing。
- [ ] 处理 padding 和 capacity。
- [ ] 验证 EP、多卡和 PD 场景。
- [ ] 理解量化、SwiGLU 和通信融合的价值。

源码与测试：

- [ ] 阅读 [`test_moe_logical_experts.py`](vllm-ascend/tests/ut/quantization/methods/test_moe_logical_experts.py)。
- [ ] 阅读 [`grouped_matmul_swiglu_quant`](vllm-ascend/csrc/gmm/grouped_matmul_swiglu_quant)。
- [ ] 阅读 [`grouped_matmul_swiglu_quant_v2`](vllm-ascend/csrc/gmm/grouped_matmul_swiglu_quant_v2)。
- [ ] 阅读 [`dispatch_ffn_combine_w4_a8`](vllm-ascend/csrc/mc2/dispatch_ffn_combine_w4_a8)。
- [ ] 阅读 [`test_grouped_matmul_swiglu_quant.py`](vllm-ascend/tests/e2e/nightly/single_node/ops/singlecard_ops/test_grouped_matmul_swiglu_quant.py)。
- [ ] 阅读 [`test_dispatch_ffn_combine_w4a8.py`](vllm-ascend/tests/e2e/nightly/single_node/ops/multicard_ops_a3/test_dispatch_ffn_combine_w4a8.py)。

阶段验收：

- [ ] 能画出 MoE 从 routing 到量化 Grouped MatMul 再到 combine 的完整数据流。
- [ ] 能解释 expert 维度、token count 和 scale shape。
- [ ] 能为 MoE 算子设计空 expert 和不均匀 routing 测试。

### 6.2 KV Cache 与 Attention 量化

- [ ] 区分权重量化、激活量化和 KV Cache 量化。
- [ ] 理解 KV Cache block layout。
- [ ] 理解 K/V scale 与 offset 的加载映射。
- [ ] 理解 C8 KV Cache。
- [ ] 理解 Prefill 写 cache 与 Decode 读 cache 的路径。
- [ ] 理解 MHA、GQA 和 MLA 的 K/V 维度差异。
- [ ] 理解部分 MLA 场景为什么只量化 K cache。
- [ ] 验证长上下文精度。
- [ ] 测量 KV Cache 容量变化。
- [ ] 测量量化/反量化对 Attention 性能的影响。

源码与测试：

- [ ] 阅读 [`kv_c8.py`](vllm-ascend/vllm_ascend/quantization/methods/kv_c8.py)。
- [ ] 阅读 [`test_kv_c8.py`](vllm-ascend/tests/ut/quantization/methods/test_kv_c8.py)。
- [ ] 阅读 [`test_kv_quant_sparse_flash_attention.py`](vllm-ascend/tests/e2e/nightly/single_node/ops/singlecard_ops/test_kv_quant_sparse_flash_attention.py)。
- [ ] 阅读 [`test_pa_kv_cache_ops.py`](vllm-ascend/tests/e2e/nightly/single_node/ops/singlecard_ops/test_pa_kv_cache_ops.py)。

---

## 第七阶段：量化模型格式

### 7.1 ModelSlim

- [ ] 使用 ModelSlim 生成 Ascend 量化模型。
- [ ] 检查模型目录中的 `quant_model_description.json`。
- [ ] 检查 weight、scale、offset 与描述文件一致。
- [ ] 使用 `--quantization ascend` 完成推理。
- [ ] 掌握模型名称映射。
- [ ] 掌握 fused module mapping。
- [ ] 能修复量化描述 key 与运行时 prefix 不一致的问题。

### 7.2 LLM-Compressor / compressed-tensors

- [ ] 理解 Hugging Face `config.json` 中的 compressed-tensors 配置。
- [ ] 理解 targets、ignore、weights 和 input activations。
- [ ] 理解 strategy、symmetric 和 dynamic 字段。
- [ ] 知道 compressed-tensors 模型通常不需要传 `--quantization ascend`。
- [ ] 理解 compressed-tensors 配置到 Ascend Scheme 的转换。
- [ ] 理解 Dense 和 MoE 配置差异。

核心源码：

- [ ] 阅读 [`compressed_tensors_config.py`](vllm-ascend/vllm_ascend/quantization/compressed_tensors_config.py)。
- [ ] 阅读 [`test_compressed_tensors_config.py`](vllm-ascend/tests/ut/quantization/test_compressed_tensors_config.py)。
- [ ] 完成一个 LLM-Compressor Dense 模型量化实验。
- [ ] 完成一个 LLM-Compressor MoE 模型量化实验。

---

## 第八阶段：图融合与自定义算子

### 8.1 Norm + Quant 图融合

典型融合：

```text
Add + RMSNorm → Dynamic Quant
```

- [ ] 理解减少中间 Tensor 回写为什么能提升性能。
- [ ] 理解 Torch FX/Inductor graph pattern 匹配。
- [ ] 理解 replacement 必须保持 shape、dtype 和输出顺序。
- [ ] 理解静态、动态和 MX 动态量化使用不同融合算子。
- [ ] 理解 SP 场景中的 all-gather/unpad。
- [ ] 理解部分 W4A4 方案为什么禁止该融合。
- [ ] 学会检查 pattern 匹配次数。
- [ ] 验证融合前后精度。
- [ ] 验证融合前后性能。

源码与测试：

- [ ] 阅读 [`norm_quant_fusion_pass.py`](vllm-ascend/vllm_ascend/compilation/passes/norm_quant_fusion_pass.py)。
- [ ] 阅读 [`test_norm_quant_fusion.py`](vllm-ascend/tests/e2e/pull_request/one_card/compile/test_norm_quant_fusion.py)。
- [ ] 阅读 [`test_graphex_norm_quant_fusion.py`](vllm-ascend/tests/e2e/pull_request/one_card/compile/test_graphex_norm_quant_fusion.py)。

### 8.2 Ascend C 自定义量化算子

- [ ] 理解 Host 侧算子定义。
- [ ] 理解 shape inference。
- [ ] 理解 tiling。
- [ ] 理解 Kernel 侧 GM/L1/UB/L0 数据搬运。
- [ ] 理解多核任务划分。
- [ ] 理解尾块处理。
- [ ] 理解 MatMul、反量化、SwiGLU 和再量化的流水融合。
- [ ] 理解 tiling key 如何选择 kernel 路径。
- [ ] 理解 INT4 unpack。
- [ ] 理解 scale 读取和累加 dtype。
- [ ] 理解 Torch adapter、ACLNN API 与 kernel 的连接。
- [ ] 能编写单算子精度测试。
- [ ] 能编写单算子性能测试。

推荐阅读顺序：

1. [ ] `op_host/*_def.cpp`
2. [ ] `op_host/*_infershape.cpp`
3. [ ] `op_host/*_tiling.cpp`
4. [ ] `op_kernel/*.cpp`
5. [ ] `op_kernel/*.h`
6. [ ] `*_torch_adpt.h`
7. [ ] 对应 `tests/e2e/nightly/.../ops` 测试

实践任务：

- [ ] 选择 `grouped_matmul_swiglu_quant`，画出 Host 到 Kernel 的调用链。
- [ ] 找到输入、输出、workspace 和 tiling 参数。
- [ ] 分析一个 tiling key 的触发条件。
- [ ] 运行或阅读对应单算子测试并记录 golden 计算方法。

---

## 第九阶段：精度与性能验收

### 9.1 单算子精度

- [ ] 使用 FP32/FP16 PyTorch 实现作为 golden。
- [ ] 固定随机种子。
- [ ] 覆盖正常分布。
- [ ] 覆盖异常值。
- [ ] 覆盖全零。
- [ ] 覆盖极值。
- [ ] 覆盖小 shape。
- [ ] 覆盖典型模型 shape。
- [ ] 覆盖非对齐 shape。
- [ ] 比较最大绝对误差。
- [ ] 比较相对误差。
- [ ] 比较 MSE。
- [ ] 比较余弦相似度。
- [ ] 单独检查 scale。
- [ ] 单独检查量化整数值。
- [ ] 覆盖 bias/no-bias。
- [ ] 覆盖不同 group size。
- [ ] MoE 覆盖空 expert 和不均匀 routing。

### 9.2 模型精度

- [ ] 使用浮点模型作为 baseline。
- [ ] 固定 tokenizer。
- [ ] 固定 prompt。
- [ ] 固定 sampling 参数。
- [ ] 固定随机种子。
- [ ] 首先进行 greedy 输出比较。
- [ ] 测试 perplexity 或业务指标。
- [ ] 比较 token-level logits。
- [ ] 分层采集输出，定位首个误差突增层。
- [ ] 测试短输入。
- [ ] 测试长输入。
- [ ] 测试长 Decode。
- [ ] 对敏感层尝试 FLOAT/W8A16 回退。
- [ ] KV Cache 量化测试长上下文累积误差。

精度报告模板：

```text
模型：
数据集：
样本数量：
输入长度分布：
输出长度：
浮点基线：
量化方案：
回退层：
PPL/任务指标：
Logits 余弦相似度：
最大误差层：
生成样例差异：
是否达到验收门槛：
```

### 9.3 性能与显存

- [ ] 记录模型加载时间。
- [ ] 记录权重显存。
- [ ] 记录 KV Cache 容量。
- [ ] 记录峰值显存。
- [ ] 记录 TTFT。
- [ ] 记录 TPOT。
- [ ] 记录单请求吞吐。
- [ ] 记录高并发总吞吐。
- [ ] 记录 Prefill tokens/s。
- [ ] 记录 Decode tokens/s。
- [ ] 记录主要算子耗时和调用次数。
- [ ] 检查 CPU↔NPU 同步。
- [ ] 对比 ACL Graph 开启与关闭。
- [ ] 同时记录输出质量，避免只追求性能。

性能报告必须包含：

```text
设备型号：
驱动/固件：
CANN：
torch_npu：
vLLM：
vLLM Ascend：
模型：
量化方案：
卡数：
TP/DP/EP：
输入长度：
输出长度：
并发数：
Warm-up 次数：
测试次数：
P50/P90/P99：
```

### 9.4 测试与代码质量

- [ ] 算法注册测试。
- [ ] 配置解析测试。
- [ ] prefix mapping 测试。
- [ ] packed module mapping 测试。
- [ ] `create_weights` shape/dtype 测试。
- [ ] 权重加载与后处理测试。
- [ ] `apply` 数值测试。
- [ ] 不支持设备/dtype 的异常测试。
- [ ] FLOAT 回退测试。
- [ ] 单卡 E2E。
- [ ] TP/EP 多卡 E2E。
- [ ] 性能回归测试。
- [ ] 真实 NPU 验证。
- [ ] 执行 `ruff` 和格式检查。
- [ ] 执行相关 UT。
- [ ] 更新开发和用户文档。

---

## 第十阶段：新模型与新算法适配

### 10.1 为新模型适配已有量化算法

- [ ] 确认浮点模型已经在 vLLM Ascend 上正常运行。
- [ ] 确认模型使用的量化算法已有 Scheme。
- [ ] 检查模型 `model_type`。
- [ ] 检查运行时层 prefix。
- [ ] 检查量化描述文件中的权重 key。
- [ ] 检查 QKV fused mapping。
- [ ] 检查 gate/up fused mapping。
- [ ] 检查 MoE experts mapping。
- [ ] 增加必要的 prefix mapping。
- [ ] 增加必要的 packed module mapping。
- [ ] 处理模型特有的权重映射。
- [ ] 处理 FLOAT 回退层。
- [ ] 添加配置映射 UT。
- [ ] 添加权重加载 UT。
- [ ] 添加真实模型 E2E。
- [ ] 完成精度和性能验收。

### 10.2 接入新量化算法

- [ ] 定义算法 ID。
- [ ] 明确权重 dtype。
- [ ] 明确激活 dtype。
- [ ] 明确静态或动态。
- [ ] 明确 scale 粒度、shape 和 dtype。
- [ ] 明确对称/非对称。
- [ ] 明确是否需要 offset。
- [ ] 明确 Linear、MoE 和 Attention 支持范围。
- [ ] 明确目标 Ascend 型号。
- [ ] 明确最低 CANN/`torch_npu` 版本。
- [ ] 在 `quantization/methods/` 新建算法文件。
- [ ] 继承对应 Ascend Scheme。
- [ ] 使用 `@register_scheme` 注册。
- [ ] 实现 `create_weights`。
- [ ] 实现 `process_weights_after_loading`。
- [ ] 实现 `apply`。
- [ ] 接入所需 NPU 算子。
- [ ] 为 ModelSlim 增加算法描述支持。
- [ ] 或为 compressed-tensors 增加配置转换。
- [ ] 处理 fused QKV。
- [ ] 处理 gate/up。
- [ ] 处理 MoE experts。
- [ ] 处理 TP/EP。
- [ ] 处理 FLOAT/高精度回退。
- [ ] 添加配置解析 UT。
- [ ] 添加 Linear/MoE 数值 UT。
- [ ] 添加 NPU 单算子测试。
- [ ] 添加真实模型 E2E。
- [ ] 完成 BF16 对照精度测试。
- [ ] 完成量化前后性能与显存测试。
- [ ] 更新支持矩阵和文档。

新算法设计记录：

```text
算法 ID：
目标设备：
Weight dtype：
Activation dtype：
Weight granularity：
Activation granularity：
Scale dtype/shape：
Offset：
Static/Dynamic：
支持层：
回退策略：
所需 NPU 算子：
软件版本要求：
精度门槛：
性能目标：
```

---

## 推荐执行顺序

```text
W8A8 Dynamic Linear
→ W8A8 Dynamic MoE
→ W8A8 Static
→ W8A16
→ W4A16
→ W4A8
→ KV C8
→ FP8/MXFP
→ W4A4/FlatQuant
→ 图融合
→ Ascend C 自定义算子
```

## 建议的阶段性实践项目

### 项目 1：手写量化实验

- [ ] 实现对称/非对称 INT8。
- [ ] 实现 Per-Tensor/Per-Channel/Per-Token。
- [ ] 输出误差对比报告。

### 项目 2：跟踪一个 W8A8 Dynamic Linear

- [ ] 选择真实模型和具体层。
- [ ] 从配置跟踪到 Scheme。
- [ ] 从 Scheme 跟踪到 NPU 算子。
- [ ] 记录各参数 shape/dtype。
- [ ] 输出调用链文档。

### 项目 3：完成一个小型源码改动

可选方向：

- [ ] 改进缺失量化配置的错误信息。
- [ ] 增加一个模型 prefix mapping 和回归测试。
- [ ] 增加 W8A8 Dynamic 边界 shape 测试。
- [ ] 增加敏感层 FLOAT 回退规则。
- [ ] 增加量化配置校验器。

### 项目 4：MoE 或 KV Cache 专项

- [ ] 完成一个 MoE 量化模型的调用链与性能分析。
- [ ] 或完成一个 C8 KV Cache 长上下文精度与容量分析。

---

## 毕业项目

推荐任务：为一个已经支持的浮点模型完整增加 W8A8 Dynamic 量化支持。

交付物：

- [ ] 量化方案说明。
- [ ] 校准数据选择依据。
- [ ] 量化模型及配置文件。
- [ ] 模型 prefix/packed mapping。
- [ ] Linear Scheme 适配。
- [ ] 如适用，MoE Scheme 适配。
- [ ] 权重 shape/dtype 清单。
- [ ] 单算子 golden 测试。
- [ ] 配置和权重加载 UT。
- [ ] 单卡 E2E。
- [ ] 多卡 TP/EP E2E。
- [ ] BF16 与量化精度报告。
- [ ] TTFT、TPOT、吞吐与显存报告。
- [ ] 已知限制。
- [ ] 敏感层回退方案。
- [ ] 支持设备和软件版本矩阵。
- [ ] 开发文档。
- [ ] 用户使用文档。

毕业验收：

- [ ] 能独立解释量化算法原理。
- [ ] 能独立定位配置、权重、scale、shape、dtype 和算子问题。
- [ ] 能完成新模型量化适配。
- [ ] 能完成一个新 Scheme 的基本实现。
- [ ] 能用数据证明精度、显存和性能是否达标。

---

## 问题与知识沉淀

建议每解决一个问题增加一条记录。

| 日期 | 问题 | 根因 | 解决方案 | 关联源码/测试 |
| --- | --- | --- | --- | --- |
| | | | | |

## 量化需求评审模板

```text
模型：
模型类型（Dense/MoE/Multi-modal）：
目标 Ascend 设备：
软件版本：
目标量化算法：
量化工具：
权重量化：
激活量化：
KV Cache 量化：
目标精度：
目标显存：
目标 TTFT：
目标 TPOT：
目标吞吐：
部署形态：
TP/DP/EP：
Prefill/Decode：
需要适配的层：
需要新增的算子：
已知敏感层：
回退策略：
测试数据集：
验收标准：
风险：
```

