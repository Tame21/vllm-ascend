# Qwen3.5 dense LoRA and Magistral graph/AOT patches for vLLM 0.25.1

These worker patches replace `lora_1.patch`, `patch_qwen3_5_dense_lora.py`
and `patch_lora_acl_graph_v23.py` from the external v0.23 workaround.
Do not install the external scripts alongside these patches.

The worker loader imports both modules only for vLLM 0.25.1 on non-310P
devices. The dense-kernel patch is enabled only for `qwen3_5_text`; the
graph/AOT patch supports both Qwen3.5 and the Mistral3-based
`mistralai/Magistral-Small-2509`. Runs without LoRA and unrelated models retain
the original implementations.

## Enable

Use the normal LoRA launch command; no feature flag, environment variable or
extra compilation configuration is required:

```bash
vllm serve /path/to/Qwen3.5-dense \
  --enable-lora \
  --max-lora-rank 16 \
  --lora-modules adapter=/path/to/adapter
```

For Magistral-Small-2509, use the same graph/LoRA settings with the Magistral
model and its language-model adapter:

```bash
VLLM_USE_AOT_COMPILE=1 vllm serve mistralai/Magistral-Small-2509 \
  --enable-lora \
  --max-lora-rank 64 \
  --lora-modules adapter=/path/to/adapter
```

Restart all workers after updating the files. The platform patch selects the
compatible graph backend automatically. No external `install(register_func)`
call is needed.

## What changes

- `patch_qwen3_5_dense_lora.py`: pads non-block-aligned LoRA-A ranks and
  temporary shrink outputs, and pads narrow or unaligned packed LoRA-B slices
  and temporary expand outputs before invoking the native Ascend kernels. It
  also preserves missing `in_proj_qkvz` LoRA groups when unpacking a partial
  adapter, and adds language prefixes only when they resolve to an actual
  module.
- `patch_lora_acl_graph.py`: compiles separate base and LoRA callables,
  separates FULL graph events/handles/workspaces/attention parameters by
  `BatchDescriptor`, keeps dummy LoRA counts consistent with capture keys,
  tracks base/adapter state for decode, and excludes Qwen GDN metadata from FIA
  replay without mutating the shared context.
  With `VLLM_USE_AOT_COMPILE=1`, the LoRA, dynamic base, and single-token base
  callables are compiled and cached as independent AOT artifacts. A runtime
  dispatcher selects the matching artifact before ACL graph capture or replay.

The second module is inactive with `--enforce-eager` or no CUDA graph mode.
The first module still provides LoRA-A/LoRA-B and name/metadata fixes in eager
mode.

## Scope and verification

This is a local compatibility patch for the Ascend `releases/v0.25.1rc`
release lane and vLLM 0.25.1, not a claim of upstream or NPU validation. The
intended initial deployment is A2/A3, runner v1, Qwen3.5 language-model LoRA
with configured rank 8 or 16, or Magistral-Small-2509 language-model LoRA with
rank <= 64, with optional MTP speculative decoding. MoE and 310P are outside
this patch's scope. Non-MTP speculative decoding, runner v2, unsupported rank,
context parallelism, microbatching and sleep mode are rejected for matching
deployments. Non-aligned shrink ranks and narrow or unaligned expand slices
allocate padded temporary weights and outputs. Graph isolation also disables
npugraph_ex automatically. Fully sharded LoRA keeps the native vLLM
tensor-parallel sharding and collective paths. Vision/connector adapters are
outside the supported scope.

Before deployment, compare eager and graph outputs for the base model and
each adapter; alternate base/adapter requests at the same token count; test
mixed base/adapter batches, prefill/decode, rank greater than a packed slice
width, and your actual TP configuration. Check that adapters change outputs
and that returning to the base model restores the base result.
