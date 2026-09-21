# Separate LoRA AOT extension for vLLM 0.25.1

`patch_lora_aot.py` contains the LoRA/base/base_one AOT artifact splitting that
previously lived in `patch_lora_acl_graph.py`. This is a code-organization change:
the default three-variant behavior and artifact suffixes are unchanged. It does
not implement or validate a new shared base/LoRA tensor-state graph.

## Responsibilities and load order

The worker loader imports the following modules, in order, on the v0.25.1
non-310P release lane:

1. `patch_qwen3_5_dense_lora`: Qwen LoRA-B and module-name compatibility.
2. `patch_lora_acl_graph`: ordinary compilation callables and routing, dynamic
   input marking, ACL graph parameters, capture bookkeeping, LoRA state, and
   Qwen FIA/GDN metadata filtering.
3. `patch_lora_aot`: AOT compilation, dispatch, cache saving, and cache loading
   for the graph patch's three callables, unless
   `VLLM_ASCEND_ENABLE_LORA_AOT=0` is set before startup.

The dependency is one-way: the AOT extension imports the graph patch; the graph
patch never imports the AOT extension. Load both before constructing model
wrappers. All workers must be restarted after changing the patch set.

The AOT extension wraps the existing wrapper initializer to install an
instance-local AOT saver, enables the AOT context while creating the two base
callables, and replaces these entry points:

- `TorchCompileWithNoGuardsWrapper.aot_compile`
- `vllm.compilation.decorators._try_load_aot_compiled_fn`

`_AOTVariantDispatcher` is stored in `model.aot_compiled_fn`, so vLLM's direct
AOT fast path still selects the current request's variant. It lazily compiles
missing variants and reuses partially populated caches. Cache files remain
`model.lora`, `model.base`, and `model.base_one` under vLLM's normal rank-specific
AOT cache directory. Source validation may cause existing artifacts to be
recompiled after this refactor.

## Run without the AOT extension

Set `VLLM_ASCEND_ENABLE_LORA_AOT=0` before launching the server to skip the AOT
extension and explicitly allow vLLM's original single-artifact AOT path. The
default is `1`, preserving three-variant AOT. No files need to be removed.
This is a startup-only switch: restart all workers between comparisons; it
does not uninstall hooks from an already running process.

| AOT | `cudagraph_specialize_lora` | `VLLM_ASCEND_ENABLE_LORA_AOT` | Behavior |
| --- | --- | --- | --- |
| `0` | `true` | `0` or `1` | Ordinary three-callable compilation and ACL graph isolation |
| `1` | `true` | `1` (default) | Independent LoRA/base/base_one AOT artifacts |
| `1` | `true` | `0` | Original single AOT artifact; ACL graph isolation retained; known AOT routing risk |
| `1` | `false` | `0` or `1` | Original single AOT entry point; LoRA graph specialization disabled |

For a native-AOT performance comparison with the graph fixes retained, merge
the compilation field into your existing configuration:

```bash
export VLLM_USE_AOT_COMPILE=1
export VLLM_ASCEND_ENABLE_LORA_AOT=0
# Use a fresh, persistent cache directory, separate from the split-AOT run.
export VLLM_CACHE_ROOT=/path/to/cache/native-aot-comparison
vllm serve /path/to/model \
  --enable-lora \
  --max-lora-rank 64 \
  --lora-modules adapter=/path/to/adapter \
  --compilation-config '{"cudagraph_specialize_lora": true}'
```

For the split-AOT comparison, set `VLLM_ASCEND_ENABLE_LORA_AOT=1` and use a
different cache directory, keeping all other model/workload settings identical.
Measure cold startup and restart with populated caches separately.

Disabling this switch leaves ACL graph descriptor isolation, capture-state
synchronization, base warmup, ordinary compilation routing, FIA/GDN metadata
filtering, and the dense/Mamba patches in place. It does not change the gated
RMSNorm AOT fix in `ops/layernorm.py` or the global `VLLM_USE_AOT_COMPILE` value.

The native-AOT mode is an intentionally uncorrected performance baseline, not
a shared-graph correctness fix. AOT calls bypass ordinary callable dispatch;
base/adapter transitions or mixed batches may produce incorrect outputs. A
startup warning makes this risk explicit. Validate outputs alongside timings;
do not treat faster incorrect results as a valid performance improvement.
Different shapes/modes can still have multiple ACL graph captures. If the switch
is left at `1` but the extension is missing, the startup error is retained.

## Verification

- `tests/ut/patch/worker/test_patch_lora_acl_graph.py`: non-AOT routing, metadata
  fixes, startup switch/load order, explicit native-AOT opt-out, and the
  missing-extension safety check.
- `tests/ut/patch/worker/test_patch_lora_aot.py`: scoped hooks, upstream fallback,
  variant dispatch, lazy compilation, full/partial/empty cache loading, atomic
  saves, and failed-save cleanup.

NPU checks remain required: compare eager/graph outputs while alternating base
and adapter requests, test mixed batches and single-token decode, then restart
with both full and partial AOT caches and repeat the comparisons. This refactor
does not add new hardware-validation claims.
