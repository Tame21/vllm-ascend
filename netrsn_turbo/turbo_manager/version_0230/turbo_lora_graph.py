# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Install the non-test changes from lora_1.patch and lora_2.patch."""

import os


def apply_lora_graph_patch():
    # Keep heavy imports inside the gate, after vllm and vllm_ascend load.
    import torch

    from netrsn_turbo.turbo_manager.turbo_utils import TurboManager
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
    from vllm_ascend.compilation import acl_graph
    from vllm_ascend.device_allocator.sleep_mem_optimized import (
        AclGraphSleepWakeupManager,
    )

    from netrsn_turbo.turbo.version_0230.vllm_ascend.attention.attention_v1 import (
        _wrap_update_graph_params,
    )
    from netrsn_turbo.turbo.version_0230.vllm_ascend.compilation import (
        acl_graph as graph_patch,
    )
    from netrsn_turbo.turbo.version_0230.vllm_ascend.device_allocator import (
        sleep_mem_optimized as sleep_patch,
    )

    if getattr(acl_graph, "_netrsn_lora_graph_patch_applied", False):
        return
    if any(
        params is not None
        for params in (
            acl_graph._graph_params,
            acl_graph._draft_graph_params,
            acl_graph._draft_graph_prefill_params,
        )
    ):
        raise RuntimeError(
            "LoRA graph patches must be applied before graph parameters are "
            "initialized. Restart the worker and import turbo_lora_graph "
            "before model-runner initialization."
        )

    # These are NEW symbols: attach directly. Registering a missing attribute
    # through a propagation engine may try to propagate the identity of None.
    acl_graph.GraphParamsByLoRA = graph_patch.GraphParamsByLoRA
    acl_graph._new_graph_params = graph_patch._new_graph_params
    acl_graph._select_graph_params = graph_patch._select_graph_params
    acl_graph.iter_graph_params = graph_patch.iter_graph_params

    # Preserve GraphParams class identity and update its nullable annotation.
    workspace_dict_type = dict[int, torch.Tensor | None]
    acl_graph.GraphParams.__annotations__["workspaces"] = workspace_dict_type
    acl_graph.GraphParams.__dataclass_fields__["workspaces"].type = (
        workspace_dict_type
    )

    # Propagation is essential: attention/model_runner/spec_decode modules use
    # `from ...acl_graph import get_* / set_* / update_*` before this import.
    for name in (
        "set_graph_params",
        "get_graph_params",
        "update_graph_params_workspaces",
        "set_draft_graph_params",
        "get_draft_graph_params",
        "update_draft_graph_params_workspaces",
        "set_draft_graph_prefill_params",
        "get_draft_graph_prefill_params",
        "update_draft_graph_prefill_params_workspaces",
        "weak_ref_workspaces",
    ):
        TurboManager.register_patch(
            f"vllm_ascend.compilation.acl_graph.{name}",
            getattr(graph_patch, name),
        )
    TurboManager.apply_patches()

    # Preserve descriptors explicitly, as in patch_mesim.md. These methods
    # are resolved on their existing classes by the runtime callers.
    AscendAttentionBackendImpl.update_graph_params = staticmethod(
        _wrap_update_graph_params(AscendAttentionBackendImpl.update_graph_params)
    )
    AclGraphSleepWakeupManager.clear_all_attention_workspaces = classmethod(
        sleep_patch.clear_all_attention_workspaces
    )
    AclGraphSleepWakeupManager.reset_all_graph_params = classmethod(
        sleep_patch.reset_all_graph_params
    )
    acl_graph._netrsn_lora_graph_patch_applied = True


# Same automatic activation policy as the example in patch_mesim.md.
# Explicit apply_lora_graph_patch() is available to an existing LoRA gate.
if os.getenv("ADAPTATION_PKG_ID", ""):
    from netrsn_turbo.coresdk.common import get_spu_n_card_type

    if get_spu_n_card_type() == "910B":
        apply_lora_graph_patch()
