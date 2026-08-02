"""Install the external vLLM/vLLM-Ascend v0.23 monkey patches.

The mirrored files under ``monkey_patch/vllm`` and
``monkey_patch/vllm-ascend`` contain code implementations only. This file is
the single patch entrypoint:

* existing functions: ``register_func(original, replacement)``;
* new functions, methods, classes, or decorated ops: direct assignment.
"""

import importlib
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
from vllm_ascend.lora.punica_npu import PunicaWrapperNPU
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


RegisterFunc = Callable[[Callable[..., Any], Callable[..., Any]], Any]
_PATCH_ROOT = Path(__file__).resolve().parent
_APPLIED = False


def _load_code_module(relative_path: str, module_name: str) -> ModuleType:
    cached_module = sys.modules.get(module_name)
    if cached_module is not None:
        return cached_module

    source_path = _PATCH_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load patch code from {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_all_code_modules() -> dict[str, ModuleType]:
    code_files = {
        "punica": "vllm-ascend/vllm_ascend/lora/punica_npu.py",
        "attention": "vllm-ascend/vllm_ascend/attention/attention_v1.py",
        "acl_graph": "vllm-ascend/vllm_ascend/compilation/acl_graph.py",
        "npu_model_runner": (
            "vllm-ascend/vllm_ascend/worker/model_runner_v1.py"
        ),
        "model_manager": "vllm/vllm/lora/model_manager.py",
        "worker_manager": "vllm/vllm/lora/worker_manager.py",
        "compile_wrapper": "vllm/vllm/compilation/wrapper.py",
    }
    return {
        name: _load_code_module(path, f"_project_monkey_patch_{name}")
        for name, path in code_files.items()
    }


def _assign_new_objects(code: dict[str, ModuleType]) -> None:
    punica_code = code["punica"]
    punica_source = importlib.import_module("vllm_ascend.lora.punica_npu")

    # Newly decorated custom ops are exposed on the original module by
    # assignment. Their decorators already run when the mirrored file loads.
    punica_source._lora_bmm_expand_slice = (
        punica_code.lora_bmm_expand_slice
    )
    punica_source._lora_bmm_expand_slice_fake = (
        punica_code.lora_bmm_expand_slice_fake
    )

    # New Punica methods.
    PunicaWrapperNPU.enable_compatible_lora_bmm_expand_slice = (
        punica_code.enable_compatible_lora_bmm_expand_slice
    )
    PunicaWrapperNPU._requires_compatible_lora_expand_slice = (
        punica_code.requires_compatible_lora_expand_slice
    )
    PunicaWrapperNPU._compatible_lora_bmm_expand_slice = (
        punica_code.compatible_lora_bmm_expand_slice
    )

    # New compile-wrapper methods.
    wrapper_code = code["compile_wrapper"]
    TorchCompileWithNoGuardsWrapper._external_punica_has_lora = (
        wrapper_code.punica_has_lora
    )
    TorchCompileWithNoGuardsWrapper._external_mark_variant_dynamic_inputs = (
        wrapper_code.mark_variant_dynamic_inputs
    )

    # New ACL graph helpers/classes.
    acl_code = code["acl_graph"]
    acl_source = importlib.import_module("vllm_ascend.compilation.acl_graph")
    acl_source._GraphParamStore = acl_code.GraphParamStore
    acl_source._make_graph_params = acl_code.make_graph_params

    # New attention helper.
    attention_code = code["attention"]
    attention_source = importlib.import_module(
        "vllm_ascend.attention.attention_v1"
    )
    attention_source._filter_fia_metadata = attention_code.filter_fia_metadata
    AscendAttentionBackendImpl.update_graph_params = staticmethod(
        attention_code.update_graph_params
    )

    # NPUModelRunner inherits these methods from vLLM. Adding overrides on the
    # Ascend subclass keeps the worker changes inside the vLLM-Ascend layer.
    npu_runner_code = code["npu_model_runner"]
    NPUModelRunner.maybe_dummy_run_with_lora = (
        npu_runner_code.maybe_dummy_run_with_lora
    )
    NPUModelRunner._warmup_and_capture = npu_runner_code.warmup_and_capture


def _register_existing_functions(
    register_func: RegisterFunc,
    code: dict[str, ModuleType],
) -> None:
    punica_code = code["punica"]
    register_func(
        punica_code.ORIGINAL_EXPAND_SLICE_PREFILL,
        punica_code.expand_slice_prefill,
    )
    register_func(
        punica_code.ORIGINAL_EXPAND_SLICE_DECODE,
        punica_code.expand_slice_decode,
    )
    register_func(punica_code.ORIGINAL_ADD_SHRINK, punica_code.add_shrink)
    register_func(punica_code.ORIGINAL_ADD_EXPAND, punica_code.add_expand)
    register_func(
        punica_code.ORIGINAL_ADD_LORA_EMBEDDING,
        punica_code.add_lora_embedding,
    )
    register_func(
        punica_code.ORIGINAL_ADD_LORA_LINEAR,
        punica_code.add_lora_linear,
    )
    register_func(
        punica_code.ORIGINAL_ADD_LORA_LOGITS,
        punica_code.add_lora_logits,
    )

    model_manager_code = code["model_manager"]
    register_func(model_manager_code.ORIGINAL_INIT, model_manager_code.init)

    worker_manager_code = code["worker_manager"]
    register_func(
        worker_manager_code.ORIGINAL_LOAD_ADAPTER,
        worker_manager_code.load_adapter,
    )

    wrapper_code = code["compile_wrapper"]
    register_func(wrapper_code.ORIGINAL_INIT, wrapper_code.init)
    register_func(wrapper_code.ORIGINAL_CALL, wrapper_code.call)

    acl_code = code["acl_graph"]
    register_func(
        acl_code.ORIGINAL_SET_GRAPH_PARAMS,
        acl_code.set_graph_params,
    )
    register_func(
        acl_code.ORIGINAL_SET_DRAFT_GRAPH_PARAMS,
        acl_code.set_draft_graph_params,
    )
    register_func(
        acl_code.ORIGINAL_SET_DRAFT_PREFILL_PARAMS,
        acl_code.set_draft_graph_prefill_params,
    )
    register_func(
        acl_code.ORIGINAL_WEAK_REF_WORKSPACES,
        acl_code.weak_ref_workspaces,
    )

    npu_runner_code = code["npu_model_runner"]
    register_func(
        npu_runner_code.ORIGINAL_CAPTURE_MODEL,
        npu_runner_code.capture_model,
    )


def _refresh_early_imported_aliases(code: dict[str, ModuleType]) -> None:
    """Refresh function aliases imported before register_func is called."""
    acl_code = code["acl_graph"]
    code["npu_model_runner"].assign_graph_param_functions(
        acl_code.set_graph_params,
        acl_code.set_draft_graph_params,
    )


def apply_patches(register_func: RegisterFunc) -> None:
    """Apply all v0.23 patches before model construction."""
    global _APPLIED
    if _APPLIED:
        return

    code = _load_all_code_modules()
    _assign_new_objects(code)
    _register_existing_functions(register_func, code)
    _refresh_early_imported_aliases(code)

    _APPLIED = True
