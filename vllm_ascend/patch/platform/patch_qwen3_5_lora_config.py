# SPDX-License-Identifier: Apache-2.0
"""Select the graph backend used by release-lane LoRA isolation patches."""

from functools import wraps

from vllm.logger import logger

from vllm_ascend.patch.lora_graph_config import specialize_lora, validate_config
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.utils import is_310p, vllm_version_is

_ORIGINAL_CHECK_AND_UPDATE_CONFIG = NPUPlatform.check_and_update_config.__func__


def _uses_supported_lora_graph(vllm_config) -> bool:
    return vllm_version_is("0.25.1") and not is_310p() and specialize_lora(vllm_config)


@wraps(_ORIGINAL_CHECK_AND_UPDATE_CONFIG)
def check_and_update_config(cls, vllm_config):
    if _uses_supported_lora_graph(vllm_config):
        validate_config(vllm_config)
        if vllm_config.additional_config is None:
            vllm_config.additional_config = {}
        graph_config = vllm_config.additional_config.setdefault("ascend_compilation_config", {})
        graph_config["enable_npugraph_ex"] = False
        graph_config["enable_static_kernel"] = False
        logger.info_once("Using the ACL graph backend required by the LoRA graph/AOT isolation patch")
    return _ORIGINAL_CHECK_AND_UPDATE_CONFIG(cls, vllm_config)


if not getattr(NPUPlatform, "_ascend_lora_config_patch_installed", False):
    NPUPlatform.check_and_update_config = classmethod(check_and_update_config)
    NPUPlatform._ascend_lora_config_patch_installed = True
