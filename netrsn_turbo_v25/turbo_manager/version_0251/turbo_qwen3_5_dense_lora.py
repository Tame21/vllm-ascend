# SPDX-License-Identifier: Apache-2.0

"""Install the Qwen3.5 dense LoRA runtime patch for vLLM 0.25.1."""

import os


def apply_qwen3_5_dense_lora_patch() -> None:
    # Resolve the real classes first so repeated/native installs can return
    # before the custom operator in the implementation module is registered.
    from vllm.lora.model_manager import LoRAModelManager
    from vllm.lora.worker_manager import WorkerLoRAManager
    from vllm_ascend.attention.attention_v1 import (
        AscendAttentionBackendImpl,
    )
    from vllm_ascend.lora.punica_npu import PunicaWrapperNPU

    if getattr(PunicaWrapperNPU, "_ascend_qwen3_5_patch_installed", False):
        return
    if getattr(
        PunicaWrapperNPU,
        "_external_qwen3_5_dense_lora_patch",
        False,
    ):
        raise RuntimeError(
            "Remove the external v0.23 Qwen3.5 LoRA patch before using "
            "the v0.25.1 TurboManager patch"
        )

    from netrsn_turbo.turbo.version_0251.vllm.lora import (
        model_manager as model_manager_patch,
    )
    from netrsn_turbo.turbo.version_0251.vllm.lora import (
        worker_manager as worker_manager_patch,
    )
    from netrsn_turbo.turbo.version_0251.vllm_ascend.attention import (
        attention_v1 as attention_patch,
    )
    from netrsn_turbo.turbo.version_0251.vllm_ascend.lora import (
        punica_npu as punica_patch,
    )

    PunicaWrapperNPU._expand_slice_prefill = punica_patch.wrap_expand_slice(
        PunicaWrapperNPU._expand_slice_prefill
    )
    PunicaWrapperNPU._expand_slice_decode = punica_patch.wrap_expand_slice(
        PunicaWrapperNPU._expand_slice_decode
    )
    PunicaWrapperNPU.update_metadata = punica_patch.wrap_update_metadata(
        PunicaWrapperNPU.update_metadata
    )
    for name in (
        "add_shrink",
        "add_expand",
        "add_lora_embedding",
        "add_lora_linear",
        "add_lora_logits",
    ):
        setattr(
            PunicaWrapperNPU,
            name,
            punica_patch.no_lora_guard(getattr(PunicaWrapperNPU, name)),
        )

    LoRAModelManager.__init__ = model_manager_patch.wrap_init(LoRAModelManager.__init__)
    WorkerLoRAManager._load_adapter = worker_manager_patch.wrap_load_adapter(
        WorkerLoRAManager._load_adapter
    )
    AscendAttentionBackendImpl.update_graph_params = staticmethod(
        attention_patch.wrap_update_graph_params(
            AscendAttentionBackendImpl.update_graph_params
        )
    )
    PunicaWrapperNPU._ascend_qwen3_5_patch_installed = True


if os.getenv("ADAPTATION_PKG_ID", ""):
    from netrsn_turbo.coresdk.common import get_spu_n_card_type

    if get_spu_n_card_type() == "910B":
        apply_qwen3_5_dense_lora_patch()
