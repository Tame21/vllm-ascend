# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.config import CUDAGraphMode

from vllm_ascend.patch import lora_graph_config as config_patch
from vllm_ascend.patch.platform import patch_qwen3_5_lora_config as platform_patch


def make_config(text="qwen3_5_text", outer="qwen3_5", model="model", architectures=(), multimodal=False):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model=model,
            hf_text_config=SimpleNamespace(model_type=text),
            hf_config=SimpleNamespace(model_type=outer, architectures=architectures),
            multimodal_config=object() if multimodal else None,
            enforce_eager=False,
            enable_sleep_mode=False,
        ),
        lora_config=SimpleNamespace(max_lora_rank=16, enable_tower_connector_lora=False),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL, cudagraph_specialize_lora=True),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1, decode_context_parallel_size=1, enable_dbo=False, ubatch_size=1
        ),
        use_v2_model_runner=False,
        speculative_config=None,
        additional_config={"other": True, "ascend_compilation_config": {"other_graph_option": True}},
    )


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, True),
        ({"text": "mistral", "outer": "mistral3"}, True),
        ({"text": "mistral", "outer": "mistral", "model": "mistralai/Magistral-Small-2509"}, True),
        ({"text": "mistral", "outer": "mistral", "architectures": ("MistralForCausalLM",), "multimodal": True}, True),
        ({"text": "mistral", "outer": "mistral"}, False),
        ({"text": "qwen3_5_moe_text"}, False),
        ({"text": "llama"}, False),
    ],
)
def test_model_matching_is_shared_by_platform_and_worker(kwargs, expected):
    config = make_config(**kwargs)
    with (
        patch.object(platform_patch, "is_310p", return_value=False),
        patch.object(platform_patch, "vllm_version_is", return_value=True),
    ):
        assert platform_patch._uses_supported_lora_graph(config) is expected
    assert config_patch.specialize_lora(config) is expected


@pytest.mark.parametrize("bypass", ["eager", "no_lora", "no_graph", "no_specialization", "no_model"])
def test_bypass_preserves_platform_configuration(bypass):
    config = make_config()
    if bypass == "eager":
        config.model_config.enforce_eager = True
    elif bypass == "no_lora":
        config.lora_config = None
    elif bypass == "no_graph":
        config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    elif bypass == "no_model":
        config.model_config = None
    else:
        config.compilation_config.cudagraph_specialize_lora = False
    with (
        patch.object(platform_patch, "is_310p", return_value=False),
        patch.object(platform_patch, "vllm_version_is", return_value=True),
        patch.object(platform_patch, "_ORIGINAL_CHECK_AND_UPDATE_CONFIG", return_value="original") as original,
    ):
        assert platform_patch.check_and_update_config("platform", config) == "original"
    original.assert_called_once_with("platform", config)
    assert config.additional_config["ascend_compilation_config"] == {"other_graph_option": True}


def test_platform_selects_acl_graph_before_initializing_ascend_config():
    config = make_config()

    def original(cls, config):
        assert config.additional_config == {
            "other": True,
            "ascend_compilation_config": {
                "other_graph_option": True,
                "enable_npugraph_ex": False,
                "enable_static_kernel": False,
            },
        }

    with (
        patch.object(platform_patch, "is_310p", return_value=False),
        patch.object(platform_patch, "vllm_version_is", return_value=True),
        patch.object(platform_patch, "_ORIGINAL_CHECK_AND_UPDATE_CONFIG", side_effect=original),
    ):
        platform_patch.check_and_update_config("platform", config)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        (None, "use_v2_model_runner", True, "runner v1"),
        ("lora_config", "max_lora_rank", 32, "8 or 16"),
        ("parallel_config", "prefill_context_parallel_size", 2, "context parallelism"),
        ("parallel_config", "decode_context_parallel_size", 2, "context parallelism"),
        ("parallel_config", "enable_dbo", True, "microbatching"),
        ("parallel_config", "ubatch_size", 2, "microbatching"),
        ("model_config", "enable_sleep_mode", True, "sleep mode"),
        (None, "speculative_config", SimpleNamespace(method="eagle3"), "MTP"),
    ],
)
def test_unsupported_graph_config_fails_early(section, field, value, message):
    config = make_config()
    setattr(getattr(config, section) if section else config, field, value)
    with pytest.raises(ValueError, match=message):
        config_patch.validate_config(config)
    config.model_config.enforce_eager = True
    config_patch.validate_config(config)


def test_magistral_rank_and_language_adapter_scope():
    config = make_config(text="mistral", outer="mistral3")
    config.lora_config.max_lora_rank = 64
    config.speculative_config = SimpleNamespace(method="mtp")
    config_patch.validate_config(config)
    config.lora_config.max_lora_rank = 128
    with pytest.raises(ValueError, match="<= 64"):
        config_patch.validate_config(config)
    config.lora_config.max_lora_rank = 64
    config.lora_config.enable_tower_connector_lora = True
    with pytest.raises(ValueError, match="language-model"):
        config_patch.validate_config(config)
