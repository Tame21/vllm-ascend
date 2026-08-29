# SPDX-License-Identifier: Apache-2.0

"""Install Qwen3.5 Base/LoRA compile and ACL graph isolation."""

import os


def apply_lora_acl_graph_patch() -> None:
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm_ascend.compilation import acl_graph

    if getattr(
        TorchCompileWithNoGuardsWrapper,
        "_ascend_lora_graph_patch_installed",
        False,
    ):
        return
    if getattr(
        TorchCompileWithNoGuardsWrapper,
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

    # Graph isolation depends on the Qwen3.5 LoRA wrapper flags and config
    # validation installed by this patch.
    from netrsn_turbo.turbo_manager.version_0251 import (
        turbo_qwen3_5_dense_lora,
    )

    turbo_qwen3_5_dense_lora.apply_qwen3_5_dense_lora_patch()

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

    TurboManager.register_patch(
        "vllm_ascend.compilation.acl_graph.GraphParams",
        graph_patch.LoRAGraphParams,
    )
    TurboManager.register_patch(
        "vllm_ascend.compilation.acl_graph.weak_ref_workspaces",
        graph_patch.wrap_weak_ref_workspaces(acl_graph.weak_ref_workspaces),
    )
    TurboManager.apply_patches()

    TorchCompileWithNoGuardsWrapper.__init__ = wrapper_patch.wrap_init(
        TorchCompileWithNoGuardsWrapper.__init__
    )
    TorchCompileWithNoGuardsWrapper.__call__ = wrapper_patch.wrap_call(
        TorchCompileWithNoGuardsWrapper.__call__
    )
    TorchCompileWithNoGuardsWrapper._ascend_has_lora = wrapper_patch.has_lora
    TorchCompileWithNoGuardsWrapper._ascend_mark_base_dynamic_inputs = (
        wrapper_patch.mark_base_dynamic_inputs
    )
    GPUModelRunner.maybe_dummy_run_with_lora = (
        model_runner_patch.wrap_maybe_dummy_run_with_lora(
            GPUModelRunner.maybe_dummy_run_with_lora
        )
    )
    GPUModelRunner._warmup_and_capture = model_runner_patch.wrap_warmup_and_capture(
        GPUModelRunner._warmup_and_capture
    )
    GPUModelRunner.capture_model = model_runner_patch.wrap_capture_model(
        GPUModelRunner.capture_model
    )
    TorchCompileWithNoGuardsWrapper._ascend_lora_graph_patch_installed = True


if os.getenv("ADAPTATION_PKG_ID", ""):
    from netrsn_turbo.coresdk.common import get_spu_n_card_type

    if get_spu_n_card_type() == "910B":
        apply_lora_acl_graph_patch()
