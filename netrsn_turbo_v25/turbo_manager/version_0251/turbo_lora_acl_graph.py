# SPDX-License-Identifier: Apache-2.0

"""Install Qwen3.5 Base/LoRA compile and ACL graph isolation."""

import os


def _get_patch_targets():
    from vllm.compilation import decorators
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm_ascend.compilation import acl_graph

    return decorators, TorchCompileWithNoGuardsWrapper, GPUModelRunner, acl_graph


def _patch_can_be_installed(wrapper_class, acl_graph) -> bool:
    if getattr(
        wrapper_class,
        "_ascend_lora_graph_patch_installed",
        False,
    ):
        return False
    if getattr(
        wrapper_class,
        "_external_lora_acl_graph_v23_patch",
        False,
    ):
        raise RuntimeError(
            "Remove the external v0.23 LoRA graph patch before using "
            "the v0.25.1 TurboManager patch"
        )
    if any(
        getattr(acl_graph, name, None) is not None
        for name in (
            "_graph_params",
            "_draft_graph_params",
            "_draft_graph_prefill_params",
        )
    ):
        raise RuntimeError(
            "LoRA ACL graph patches must be applied before graph "
            "parameters are initialized"
        )
    return True


def _install_dense_lora_dependency():
    # Graph isolation reuses the dense LoRA wrapper flags and validation.
    from netrsn_turbo.turbo_manager.version_0251 import (
        turbo_qwen3_5_dense_lora,
    )

    turbo_qwen3_5_dense_lora.apply_qwen3_5_dense_lora_patch()


def _get_patch_modules():
    from netrsn_turbo.turbo.version_0251.vllm.compilation import (
        wrapper as wrapper_patch,
    )
    from netrsn_turbo.turbo.version_0251.vllm.v1.worker import (
        gpu_model_runner as model_runner_patch,
    )
    from netrsn_turbo.turbo.version_0251.vllm_ascend.compilation import (
        acl_graph as graph_patch,
    )
    from netrsn_turbo.turbo_manager.turbo_utils import TurboManager

    return wrapper_patch, model_runner_patch, graph_patch, TurboManager


def _patch_acl_graph(TurboManager, graph_patch, acl_graph):
    TurboManager.register_patch(
        "vllm_ascend.compilation.acl_graph.GraphParams",
        graph_patch.LoRAGraphParams,
    )
    TurboManager.register_patch(
        "vllm_ascend.compilation.acl_graph.weak_ref_workspaces",
        graph_patch.wrap_weak_ref_workspaces(acl_graph.weak_ref_workspaces),
    )
    TurboManager.apply_patches()


def _patch_compile_wrapper(decorators, wrapper_class, wrapper_patch):
    wrapper_class.__init__ = wrapper_patch.wrap_init(wrapper_class.__init__)
    wrapper_class.__call__ = wrapper_patch.wrap_call(wrapper_class.__call__)
    wrapper_class.aot_compile = wrapper_patch.wrap_aot_compile(
        wrapper_class.aot_compile
    )
    wrapper_class._ascend_has_lora = wrapper_patch.has_lora
    wrapper_class._ascend_mark_variant_dynamic_inputs = (
        wrapper_patch.mark_variant_dynamic_inputs
    )
    decorators._try_load_aot_compiled_fn = (
        wrapper_patch.wrap_try_load_aot_compiled_fn(
            decorators._try_load_aot_compiled_fn
        )
    )


def _patch_model_runner(model_runner, model_runner_patch):
    model_runner.maybe_dummy_run_with_lora = (
        model_runner_patch.wrap_maybe_dummy_run_with_lora(
            model_runner.maybe_dummy_run_with_lora
        )
    )
    model_runner._warmup_and_capture = model_runner_patch.wrap_warmup_and_capture(
        model_runner._warmup_and_capture
    )
    model_runner.capture_model = model_runner_patch.wrap_capture_model(
        model_runner.capture_model
    )


def apply_lora_acl_graph_patch() -> None:
    targets = _get_patch_targets()
    decorators, wrapper_class, model_runner, acl_graph = targets
    if not _patch_can_be_installed(wrapper_class, acl_graph):
        return

    _install_dense_lora_dependency()
    wrapper_patch, runner_patch, graph_patch, TurboManager = _get_patch_modules()
    _patch_acl_graph(TurboManager, graph_patch, acl_graph)
    _patch_compile_wrapper(decorators, wrapper_class, wrapper_patch)
    _patch_model_runner(model_runner, runner_patch)
    wrapper_class._ascend_lora_graph_patch_installed = True


if os.getenv("ADAPTATION_PKG_ID", ""):
    from netrsn_turbo.coresdk.common import get_spu_n_card_type

    if get_spu_n_card_type() == "910B":
        apply_lora_acl_graph_patch()
