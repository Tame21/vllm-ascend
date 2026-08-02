"""Install the external vLLM/vLLM-Ascend v0.23 monkey patches."""

import vllm.envs as vllm_envs
from netrsn_turbo.turbo_manager.turbo_utils import TurboManager


_APPLIED = False


def apply_lora_patch() -> None:
    global _APPLIED
    if _APPLIED:
        return

    # v0.23 defaults this to True. It must be disabled before wrapper/model
    # construction because Base and LoRA use different compiled callables.
    vllm_envs.VLLM_USE_BYTECODE_HOOK = False

    import vllm_ascend.attention.attention_v1 as attention_source
    import vllm_ascend.compilation.acl_graph as acl_source
    import vllm_ascend.lora.punica_npu as punica_source
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
    from vllm_ascend.lora.punica_npu import PunicaWrapperNPU
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    from new_23.vllm_ascend.compilation.acl_graph import (
        GraphParamStore,
        make_graph_params,
        set_draft_graph_params,
        set_draft_graph_prefill_params,
        set_graph_params,
        weak_ref_workspaces,
    )
    from new_23.vllm.compilation.wrapper import call_with_optional_nvtx_range
    from new_23.vllm.lora.worker_manager import (
        create_lora_manager,
        create_lora_manager_lru,
        load_adapter,
    )
    from new_23.vllm_ascend.attention.attention_v1 import (
        filter_fia_metadata,
        update_graph_params,
    )
    from new_23.vllm_ascend.lora.punica_npu import (
        add_expand,
        add_lora_embedding,
        add_lora_linear,
        add_lora_logits,
        add_shrink,
        compatible_lora_bmm_expand_slice,
        enable_compatible_lora_bmm_expand_slice,
        expand_slice_decode,
        expand_slice_prefill,
        lora_bmm_expand_slice,
        lora_bmm_expand_slice_fake,
        requires_compatible_lora_expand_slice,
    )
    from new_23.vllm_ascend.worker.model_runner_v1 import (
        assign_graph_param_functions,
        capture_model,
        maybe_dummy_run_with_lora,
        warmup_and_capture,
    )

    TurboManager.register_patch(
        "vllm_ascend.lora.punica_npu.PunicaWrapperNPU.expand_slice_prefill",
        expand_slice_prefill,
    )
    TurboManager.register_patch(
        "vllm_ascend.lora.punica_npu.PunicaWrapperNPU.expand_slice_decode",
        expand_slice_decode,
    )
    TurboManager.register_patch(
        "vllm_ascend.lora.punica_npu.PunicaWrapperNPU.add_shrink",
        add_shrink,
    )
    TurboManager.register_patch(
        "vllm_ascend.lora.punica_npu.PunicaWrapperNPU.add_expand",
        add_expand,
    )
    TurboManager.register_patch(
        "vllm_ascend.lora.punica_npu.PunicaWrapperNPU.add_lora_embedding",
        add_lora_embedding,
    )
    TurboManager.register_patch(
        "vllm_ascend.lora.punica_npu.PunicaWrapperNPU.add_lora_linear",
        add_lora_linear,
    )
    TurboManager.register_patch(
        "vllm_ascend.lora.punica_npu.PunicaWrapperNPU.add_lora_logits",
        add_lora_logits,
    )
    TurboManager.register_patch(
        "vllm.lora.worker_manager.WorkerLoRAManager.create_lora_manager",
        create_lora_manager,
    )
    TurboManager.register_patch(
        "vllm.lora.worker_manager.LRUCacheWorkerLoRAManager.create_lora_manager",
        create_lora_manager_lru,
    )
    TurboManager.register_patch(
        "vllm.lora.worker_manager.WorkerLoRAManager._load_adapter",
        load_adapter,
    )
    TurboManager.register_patch(
        "vllm.compilation.wrapper.TorchCompileWithNoGuardsWrapper."
        "_call_with_optional_nvtx_range",
        call_with_optional_nvtx_range,
    )
    TurboManager.register_patch(
        "vllm_ascend.compilation.acl_graph.set_graph_params",
        set_graph_params,
    )
    TurboManager.register_patch(
        "vllm_ascend.compilation.acl_graph.set_draft_graph_params",
        set_draft_graph_params,
    )
    TurboManager.register_patch(
        "vllm_ascend.compilation.acl_graph.set_draft_graph_prefill_params",
        set_draft_graph_prefill_params,
    )
    TurboManager.register_patch(
        "vllm_ascend.compilation.acl_graph.weak_ref_workspaces",
        weak_ref_workspaces,
    )
    TurboManager.register_patch(
        "vllm_ascend.worker.model_runner_v1.NPUModelRunner.capture_model",
        capture_model,
    )

    TurboManager.apply_patches()

    punica_source._lora_bmm_expand_slice = lora_bmm_expand_slice
    punica_source._lora_bmm_expand_slice_fake = lora_bmm_expand_slice_fake
    PunicaWrapperNPU.enable_compatible_lora_bmm_expand_slice = (
        enable_compatible_lora_bmm_expand_slice
    )
    PunicaWrapperNPU._requires_compatible_lora_expand_slice = (
        requires_compatible_lora_expand_slice
    )
    PunicaWrapperNPU._compatible_lora_bmm_expand_slice = (
        compatible_lora_bmm_expand_slice
    )

    acl_source._GraphParamStore = GraphParamStore
    acl_source._make_graph_params = make_graph_params

    attention_source._filter_fia_metadata = filter_fia_metadata
    AscendAttentionBackendImpl.update_graph_params = staticmethod(
        update_graph_params
    )

    NPUModelRunner.maybe_dummy_run_with_lora = maybe_dummy_run_with_lora
    NPUModelRunner._warmup_and_capture = warmup_and_capture

    # model_runner_v1 imported these functions by name before TurboManager
    # replaced the acl_graph module attributes, so refresh those aliases.
    assign_graph_param_functions(
        set_graph_params,
        set_draft_graph_params,
    )

    _APPLIED = True


apply_lora_patch()
