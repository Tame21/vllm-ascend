# SPDX-License-Identifier: Apache-2.0
"""Select the compatible graph backend for Qwen3.5 dense LoRA."""

from functools import wraps

from vllm.logger import logger

from vllm_ascend.platform import NPUPlatform
from vllm_ascend.utils import is_310p, vllm_version_is

_ORIGINAL_CHECK_AND_UPDATE_CONFIG = NPUPlatform.check_and_update_config.__func__


def _uses_qwen3_5_dense_lora(vllm_config) -> bool:
    model_config = vllm_config.model_config
    return bool(
        vllm_version_is("0.25.1")
        and not is_310p()
        and model_config is not None
        and vllm_config.lora_config is not None
        and getattr(model_config.hf_text_config, "model_type", None) == "qwen3_5_text"
    )


@wraps(_ORIGINAL_CHECK_AND_UPDATE_CONFIG)
def check_and_update_config(cls, vllm_config):
    if _uses_qwen3_5_dense_lora(vllm_config):
        if vllm_config.additional_config is None:
            vllm_config.additional_config = {}
        graph_config = vllm_config.additional_config.setdefault("ascend_compilation_config", {})
        graph_config["enable_npugraph_ex"] = False
        graph_config["enable_static_kernel"] = False
        logger.info_once("Using the ACL graph backend required by the Qwen3.5 dense LoRA compatibility patch")
    return _ORIGINAL_CHECK_AND_UPDATE_CONFIG(cls, vllm_config)


NPUPlatform.check_and_update_config = classmethod(check_and_update_config)
