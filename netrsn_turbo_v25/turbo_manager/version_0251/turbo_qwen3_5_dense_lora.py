# SPDX-License-Identifier: Apache-2.0

"""Install the Qwen3.5 dense LoRA runtime patch for vLLM 0.25.1."""

import os


def _get_patch_targets():
    from vllm.lora.model_manager import LoRAModelManager
    from vllm.lora.worker_manager import WorkerLoRAManager
    from vllm_ascend.attention.attention_v1 import (
        AscendAttentionBackendImpl,
    )
    from vllm_ascend.lora.punica_npu import PunicaWrapperNPU

    return (
        LoRAModelManager,
        WorkerLoRAManager,
        AscendAttentionBackendImpl,
        PunicaWrapperNPU,
    )


def _patch_can_be_installed(PunicaWrapperNPU) -> bool:
    if getattr(PunicaWrapperNPU, "_ascend_qwen3_5_patch_installed", False):
        return False
    if getattr(
        PunicaWrapperNPU,
        "_external_qwen3_5_dense_lora_patch",
        False,
    ):
        raise RuntimeError(
            "Remove the external v0.23 Qwen3.5 LoRA patch before using "
            "the v0.25.1 TurboManager patch"
        )
    return True


def _get_patch_modules():
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

    return (
        model_manager_patch,
        worker_manager_patch,
        attention_patch,
        punica_patch,
    )


def _patch_punica_wrapper(PunicaWrapperNPU, punica_patch):
    PunicaWrapperNPU.__init__ = punica_patch.wrap_init(PunicaWrapperNPU.__init__)
    PunicaWrapperNPU.update_metadata = punica_patch.wrap_update_metadata(
        PunicaWrapperNPU.update_metadata
    )
    guarded_methods = (
        "add_shrink",
        "add_expand",
        "add_lora_embedding",
        "add_lora_linear",
        "add_lora_logits",
    )
    for name in guarded_methods:
        original = getattr(PunicaWrapperNPU, name)
        setattr(PunicaWrapperNPU, name, punica_patch.no_lora_guard(original))


def _patch_lora_managers(
    LoRAModelManager,
    WorkerLoRAManager,
    model_manager_patch,
    worker_manager_patch,
):
    LoRAModelManager.__init__ = model_manager_patch.wrap_init(
        LoRAModelManager.__init__
    )
    WorkerLoRAManager._load_adapter = worker_manager_patch.wrap_load_adapter(
        WorkerLoRAManager._load_adapter
    )


def _patch_attention_backend(AscendAttentionBackendImpl, attention_patch):
    wrapped = attention_patch.wrap_update_graph_params(
        AscendAttentionBackendImpl.update_graph_params
    )
    AscendAttentionBackendImpl.update_graph_params = staticmethod(wrapped)


def apply_qwen3_5_dense_lora_patch() -> None:
    targets = _get_patch_targets()
    model_manager, worker_manager, attention_backend, punica_wrapper = targets
    if not _patch_can_be_installed(punica_wrapper):
        return

    patches = _get_patch_modules()
    model_patch, worker_patch, attention_patch, punica_patch = patches
    _patch_punica_wrapper(punica_wrapper, punica_patch)
    _patch_lora_managers(
        model_manager,
        worker_manager,
        model_patch,
        worker_patch,
    )
    _patch_attention_backend(attention_backend, attention_patch)
    punica_wrapper._ascend_qwen3_5_patch_installed = True


if os.getenv("ADAPTATION_PKG_ID", ""):
    from netrsn_turbo.coresdk.common import get_spu_n_card_type

    if get_spu_n_card_type() == "910B":
        apply_qwen3_5_dense_lora_patch()
