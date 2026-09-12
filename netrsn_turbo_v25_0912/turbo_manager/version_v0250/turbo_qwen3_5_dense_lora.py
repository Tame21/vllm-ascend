# SPDX-License-Identifier: Apache-2.0

"""Install dense LoRA runtime support for Qwen3.5 and Magistral."""

from netrsn_turbo.turbo_manager.turbo_utils import TurboManager


def _register_punica_patches(punica_cls, patch) -> None:
    TurboManager.register_patch(
        "vllm_ascend.lora.punica_npu.PunicaWrapperNPU.update_metadata",
        patch.wrap_update_metadata(punica_cls.update_metadata),
    )
    for name in (
        "add_shrink",
        "add_expand",
        "add_lora_embedding",
        "add_lora_linear",
        "add_lora_logits",
    ):
        TurboManager.register_patch(
            f"vllm_ascend.lora.punica_npu.PunicaWrapperNPU.{name}",
            patch.no_lora_guard(getattr(punica_cls, name)),
        )


def _register_manager_patches(model_cls, worker_cls, model_patch, worker_patch):
    TurboManager.register_patch(
        "vllm.lora.model_manager.LoRAModelManager.__init__",
        model_patch.wrap_init(model_cls.__init__),
    )
    TurboManager.register_patch(
        "vllm.lora.worker_manager.WorkerLoRAManager._load_adapter",
        worker_patch.wrap_load_adapter(worker_cls._load_adapter),
    )


def _register_attention_patch(attention_cls, patch) -> None:
    wrapped = patch.wrap_update_graph_params(attention_cls.update_graph_params)
    TurboManager.register_patch(
        "vllm_ascend.attention.attention_v1."
        "AscendAttentionBackendImpl.update_graph_params",
        staticmethod(wrapped),
    )


def apply_dense_lora_patch() -> None:
    from netrsn_turbo.turbo.version_v0250.vllm.lora import (
        model_manager as model_manager_patch,
    )
    from netrsn_turbo.turbo.version_v0250.vllm.lora import (
        worker_manager as worker_manager_patch,
    )
    from netrsn_turbo.turbo.version_v0250.vllm_ascend.attention import (
        attention_v1 as attention_patch,
    )
    from netrsn_turbo.turbo.version_v0250.vllm_ascend.lora import (
        punica_npu as punica_patch,
    )
    from vllm.lora.model_manager import LoRAModelManager
    from vllm.lora.worker_manager import WorkerLoRAManager
    # Import ops in its intended direction before Punica imports lora.utils.
    # Otherwise lora.fused_moe -> ops.__init__ -> ops.fused_moe can request
    # sync_lora_context from the still-partial lora.fused_moe module.
    import vllm_ascend.ops  # noqa: F401

    from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
    from vllm_ascend.lora.punica_npu import PunicaWrapperNPU

    if getattr(PunicaWrapperNPU, "_ascend_dense_lora_patch_installed", False):
        return
    if getattr(PunicaWrapperNPU, "_external_qwen3_5_dense_lora_patch", False):
        raise RuntimeError(
            "Remove the external v0.23 Qwen3.5 LoRA patch before using "
            "the v0.25.1 TurboManager patch"
        )

    _register_punica_patches(PunicaWrapperNPU, punica_patch)
    _register_manager_patches(
        LoRAModelManager,
        WorkerLoRAManager,
        model_manager_patch,
        worker_manager_patch,
    )
    _register_attention_patch(AscendAttentionBackendImpl, attention_patch)
    TurboManager.apply_patches()
    PunicaWrapperNPU._ascend_dense_lora_patch_installed = True


# Compatibility for existing version initializers and explicit callers.
apply_qwen3_5_dense_lora_patch = apply_dense_lora_patch
