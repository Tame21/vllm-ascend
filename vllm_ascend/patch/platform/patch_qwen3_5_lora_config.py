# SPDX-License-Identifier: Apache-2.0
"""Select the graph backend used by release-lane LoRA isolation patches."""

from functools import wraps

from vllm.logger import logger

from vllm_ascend.platform import NPUPlatform
from vllm_ascend.utils import is_310p, vllm_version_is

_ORIGINAL_CHECK_AND_UPDATE_CONFIG = NPUPlatform.check_and_update_config.__func__


def _uses_supported_lora_graph(vllm_config) -> bool:
    model_config = vllm_config.model_config
    if not vllm_version_is("0.25.1") or is_310p() or model_config is None or vllm_config.lora_config is None:
        return False
    text_model_type = getattr(model_config.hf_text_config, "model_type", None)
    hf_config = model_config.hf_config
    is_magistral = bool(
        text_model_type == "mistral"
        and (
            getattr(hf_config, "model_type", None) == "mistral3"
            or "magistral" in str(getattr(model_config, "model", "")).lower()
        )
    )
    return bool(text_model_type == "qwen3_5_text" or is_magistral)


@wraps(_ORIGINAL_CHECK_AND_UPDATE_CONFIG)
def check_and_update_config(cls, vllm_config):
    if _uses_supported_lora_graph(vllm_config):
        if vllm_config.additional_config is None:
            vllm_config.additional_config = {}
        graph_config = vllm_config.additional_config.setdefault("ascend_compilation_config", {})
        graph_config["enable_npugraph_ex"] = False
        graph_config["enable_static_kernel"] = False
        logger.info_once("Using the ACL graph backend required by the LoRA graph/AOT isolation patch")
    return _ORIGINAL_CHECK_AND_UPDATE_CONFIG(cls, vllm_config)


NPUPlatform.check_and_update_config = classmethod(check_and_update_config)
