# Qwen3.5 dense LoRA patches for vLLM 0.25.1

These worker patches replace `lora_1.patch`, `patch_qwen3_5_dense_lora.py`
and `patch_lora_acl_graph_v23.py` from the external v0.23 workaround.
Do not install the external scripts alongside these patches.

The worker loader imports both modules only for vLLM 0.25.1 on non-310P
devices. They are automatically enabled only for `qwen3_5_text` dense models
with LoRA enabled. Other models and runs without LoRA retain the original
implementations.

## Enable

Use the normal LoRA launch command; no feature flag, environment variable or
extra compilation configuration is required:

```bash
vllm serve /path/to/Qwen3.5-dense \
  --enable-lora \
  --max-lora-rank 64 \
  --lora-modules adapter=/path/to/adapter
```

Restart all workers after updating the files. The platform patch selects the
compatible graph backend automatically. No external `install(register_func)`
call is needed.

## What changes

- `patch_qwen3_5_dense_lora.py`: uses a bounded, masked PyTorch fallback for
  language-model LoRA-B slices; adds language prefixes only when they resolve
  to an actual module; refreshes `no_lora` from CPU mapping data; filters GDN
  metadata out of target FIA replay without changing the shared context.
- `patch_lora_acl_graph.py`: compiles separate base and LoRA callables,
  separates FULL graph events/handles/workspaces/attention parameters by
  `BatchDescriptor`, and keeps dummy LoRA counts consistent with capture keys.

The second module is inactive with `--enforce-eager` or no CUDA graph mode.
The first module still provides LoRA-B and name/metadata fixes in eager mode.

## Scope and verification

This is a local compatibility patch against Ascend commit `3cb3f83` and vLLM
tag `v0.25.1`, not a claim of upstream or NPU validation. The intended initial
deployment is A2/A3, runner v1, dense Qwen3.5, language-model LoRA, rank <= 64,
with optional MTP speculative decoding. MoE and 310P are outside this patch's
scope. Non-MTP speculative decoding, runner v2, rank > 64, context parallelism,
microbatching and sleep mode are rejected for matching deployments. Graph
isolation also disables npugraph_ex automatically. Fully sharded LoRA keeps
the native vLLM tensor-parallel sharding and collective paths.
Vision/connector adapters are outside the supported scope. The fallback can
be slower than native kernels.

Before deployment, compare eager and graph outputs for the base model and
each adapter; alternate base/adapter requests at the same token count; test
mixed base/adapter batches, prefill/decode, rank greater than a packed slice
width, and your actual TP configuration. Check that adapters change outputs
and that returning to the base model restores the base result.
