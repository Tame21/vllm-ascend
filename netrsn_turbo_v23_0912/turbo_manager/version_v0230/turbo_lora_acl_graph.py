# SPDX-License-Identifier: Apache-2.0

"""Install Base/LoRA compile and ACL graph isolation."""

import os

from netrsn_turbo.coresdk.common import get_spu_n_card_type
from netrsn_turbo.turbo_manager.turbo_utils import TurboManager


def _validate_patch_order(wrapper_cls, acl_graph) -> bool:
    if getattr(wrapper_cls, "_ascend_lora_graph_patch_installed", False):
        return False
    if getattr(wrapper_cls, "_external_lora_acl_graph_v23_patch", False):
        raise RuntimeError(
            "Remove the external v0.23 LoRA graph patch before using "
            "the v0.23.0 TurboManager patch"
        )
    graph_param_names = (
        "_graph_params",
        "_draft_graph_params",
        "_draft_graph_prefill_params",
    )
    if any(getattr(acl_graph, name, None) is not None for name in graph_param_names):
        raise RuntimeError(
            "LoRA ACL graph patches must be applied before graph "
            "parameters are initialized"
        )
    return True


def _register_graph_patches(acl_graph, graph_patch) -> None:
    TurboManager.register_patch(
        "vllm_ascend.compilation.acl_graph.GraphParams",
        graph_patch.LoRAGraphParams,
    )
    TurboManager.register_patch(
        "vllm_ascend.compilation.acl_graph.weak_ref_workspaces",
        graph_patch.wrap_weak_ref_workspaces(acl_graph.weak_ref_workspaces),
    )


def _register_compile_patches(wrapper_cls, decorators, wrapper_patch) -> None:
    methods = {
        "__init__": wrapper_patch.wrap_init(wrapper_cls.__init__),
        "__call__": wrapper_patch.wrap_call(wrapper_cls.__call__),
        "aot_compile": wrapper_patch.wrap_aot_compile(wrapper_cls.aot_compile),
    }
    for name, replacement in methods.items():
        TurboManager.register_patch(
            f"vllm.compilation.wrapper.TorchCompileWithNoGuardsWrapper.{name}",
            replacement,
        )
    TurboManager.register_patch(
        "vllm.compilation.decorators._try_load_aot_compiled_fn",
        wrapper_patch.wrap_try_load_aot_compiled_fn(
            decorators._try_load_aot_compiled_fn
        ),
    )


def _register_runner_patches(runner_cls, runner_patch) -> None:
    methods = {
        "maybe_dummy_run_with_lora": runner_patch.wrap_maybe_dummy_run_with_lora,
        "_warmup_and_capture": runner_patch.wrap_warmup_and_capture,
        "capture_model": runner_patch.wrap_capture_model,
    }
    for name, wrap in methods.items():
        TurboManager.register_patch(
            f"vllm.v1.worker.gpu_model_runner.GPUModelRunner.{name}",
            wrap(getattr(runner_cls, name)),
        )


def apply_lora_acl_graph_patch() -> None:
    from netrsn_turbo.turbo.version_v0230.vllm.compilation import (
        wrapper as wrapper_patch,
    )
    from netrsn_turbo.turbo.version_v0230.vllm.v1.worker import (
        gpu_model_runner as model_runner_patch,
    )
    from netrsn_turbo.turbo.version_v0230.vllm_ascend.compilation import (
        acl_graph as graph_patch,
    )
    from netrsn_turbo.turbo_manager.version_v0230.turbo_qwen3_5_dense_lora import (
        apply_dense_lora_patch,
    )
    from vllm.compilation import decorators
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm_ascend.compilation import acl_graph

    if not _validate_patch_order(TorchCompileWithNoGuardsWrapper, acl_graph):
        return
    apply_dense_lora_patch()
    _register_graph_patches(acl_graph, graph_patch)
    _register_compile_patches(
        TorchCompileWithNoGuardsWrapper,
        decorators,
        wrapper_patch,
    )
    _register_runner_patches(GPUModelRunner, model_runner_patch)
    TurboManager.apply_patches()
    TorchCompileWithNoGuardsWrapper._ascend_lora_graph_patch_installed = True


_is_lora_env = bool(os.getenv("ADAPTATION_PKG_ID", ""))
if _is_lora_env and get_spu_n_card_type() == "910B":
    apply_lora_acl_graph_patch()

