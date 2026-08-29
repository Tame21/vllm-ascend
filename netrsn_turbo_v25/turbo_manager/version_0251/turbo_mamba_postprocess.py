# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Install the vLLM 0.25 Mamba postprocess compatibility patch."""


def apply_mamba_postprocess_patch() -> None:
    # Keep heavy imports here, after vllm and vllm_ascend have loaded.
    from netrsn_turbo.turbo.version_0251.vllm_ascend.ops.triton.mamba import (
        postprocess as mamba_postprocess_patch,
    )
    from netrsn_turbo.turbo_manager.turbo_utils import TurboManager
    from vllm_ascend.patch.worker import patch_mamba_utils

    if getattr(
        patch_mamba_utils,
        "_netrsn_mamba_postprocess_patch_applied",
        False,
    ):
        return

    # Replace the module binding introduced by the changed import in
    # patch_mamba_utils. TurboManager also updates the reference that this
    # module has already assigned to vllm.v1.worker.mamba_utils.
    TurboManager.register_patch(
        "vllm_ascend.patch.worker.patch_mamba_utils.postprocess_mamba_fused_kernel",
        mamba_postprocess_patch.postprocess_mamba_fused_kernel,
    )
    TurboManager.apply_patches()
    patch_mamba_utils._netrsn_mamba_postprocess_patch_applied = True


apply_mamba_postprocess_patch()
