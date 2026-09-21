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
   for the graph patch's three callables.

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

To leave the extension out of a deployment, omit `patch_lora_aot.py` and remove
only its import from `vllm_ascend/patch/worker/__init__.py`. Keep the graph patch,
the dense patch, the platform configuration patch, and the Mamba compatibility
patches. Do not replace the entire worker loader with an older copy.

| AOT | `cudagraph_specialize_lora` | AOT extension | Behavior |
| --- | --- | --- | --- |
| `0` | `true` | Optional | Ordinary three-callable compilation and ACL graph isolation |
| `1` | `true` | Loaded | Independent LoRA/base/base_one AOT artifacts |
| `1` | `true` | Omitted | Explicit startup error; no unsafe single-artifact fallback |
| `1` | `false` | Optional | Delegate to vLLM's original single AOT entry point |

For the last row, merge this field into the existing compilation configuration:

```bash
VLLM_USE_AOT_COMPILE=1 vllm serve /path/to/model \
  --enable-lora \
  --max-lora-rank 64 \
  --lora-modules adapter=/path/to/adapter \
  --compilation-config '{"cudagraph_specialize_lora": false}'
```

This removes the patch's LoRA-state AOT splitting, not all shape-specific ACL
graph captures. It also does not establish that the native shared graph safely
handles every base/adapter transition: Punica's Python control flow and changing
prefill metadata still require runtime validation. Never enable specialized
base/LoRA callables with a single un-dispatched AOT artifact.

## Verification

- `tests/ut/patch/worker/test_patch_lora_acl_graph.py`: non-AOT routing, metadata
  fixes, and the missing-extension safety check.
- `tests/ut/patch/worker/test_patch_lora_aot.py`: scoped hooks, upstream fallback,
  variant dispatch, lazy compilation, full/partial/empty cache loading, atomic
  saves, and failed-save cleanup.

NPU checks remain required: compare eager/graph outputs while alternating base
and adapter requests, test mixed batches and single-token decode, then restart
with both full and partial AOT caches and repeat the comparisons. This refactor
does not add new hardware-validation claims.
