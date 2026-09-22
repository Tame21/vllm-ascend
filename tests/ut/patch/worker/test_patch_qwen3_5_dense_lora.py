# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.patch.worker import patch_qwen3_5_dense_lora as dense


def test_install_only_patches_adapter_names_and_preserves_punica(monkeypatch):
    class Manager:
        pass

    worker_manager = SimpleNamespace(_load_adapter=MagicMock())
    punica = SimpleNamespace(
        _shrink_prefill=MagicMock(),
        _shrink_decode=MagicMock(),
        _expand_slice_prefill=MagicMock(),
        _expand_slice_decode=MagicMock(),
    )
    original_punica = vars(punica).copy()
    monkeypatch.setattr(dense, "LoRAModelManager", Manager)
    monkeypatch.setattr(dense, "WorkerLoRAManager", worker_manager)
    monkeypatch.setattr(dense, "PunicaWrapperNPU", punica)

    dense._install()
    dense._install()

    assert vars(punica) == original_punica
    assert Manager.__init__ is dense._manager_init
    assert worker_manager._load_adapter is dense._load_adapter
    assert Manager._ascend_qwen3_5_lora_names_patch_installed is True


@pytest.mark.parametrize("is_qwen", [False, True])
@pytest.mark.parametrize("specialize", [False, True])
def test_manager_enables_name_remapping_without_marking_kernel_wrappers(is_qwen, specialize):
    language_wrapper = SimpleNamespace()
    manager = SimpleNamespace(
        supports_mm=True,
        mm_mapping=SimpleNamespace(language_model=["language_model"]),
        punica_wrapper_mapping={"language_model": language_wrapper},
    )
    with (
        patch.object(dense, "is_qwen3_5", return_value=is_qwen),
        patch.object(dense, "specialize_lora", return_value=specialize),
        patch.object(dense, "validate_config") as validate,
        patch.object(dense, "_ORIGINAL_MANAGER_INIT") as original,
    ):
        dense._manager_init(manager, "model", 4, 32, 32000, "lora_config", "npu", "config")

    original.assert_called_once_with(manager, "model", 4, 32, 32000, "lora_config", "npu", "config")
    assert manager._ascend_qwen3_5_lora is (is_qwen and specialize)
    assert validate.call_count == int(is_qwen and specialize)
    assert vars(language_wrapper) == {}


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("multimodal", [False, True])
def test_adapter_load_preserves_name_remapping_scope(enabled, multimodal):
    key = "model.layers.0.in_proj_a"
    weights = object()
    adapter = SimpleNamespace(loras={key: weights})
    manager = SimpleNamespace(
        _ascend_qwen3_5_lora=enabled,
        supports_mm=multimodal,
        modules={"language_model.model.layers.0.in_proj_ba": object()},
        packed_modules_mapping={"in_proj_ba": ["in_proj_b", "in_proj_a"]},
        mm_mapping=SimpleNamespace(language_model=["language_model"]),
    )
    worker = SimpleNamespace(_adapter_manager=manager)
    with patch.object(dense, "_ORIGINAL_LOAD_ADAPTER", return_value=adapter) as original:
        assert dense._load_adapter(worker, "request") is adapter
    original.assert_called_once_with(worker, "request")
    expected_key = f"language_model.{key}" if enabled and multimodal else key
    assert adapter.loras == {expected_key: weights}


def test_module_remapping_resolves_packed_children_without_double_prefix():
    key = "model.layers.0.in_proj_a"
    full_key = "language_model." + key
    weights = object()
    modules = ["language_model.model.layers.0.in_proj_ba"]
    packed = {"in_proj_ba": ["in_proj_b", "in_proj_a"]}
    assert dense._remap_lora_keys({key: weights}, modules, packed, ["language_model"]) == {full_key: weights}
    assert dense._remap_lora_keys({full_key: weights}, modules, packed, ["language_model"]) == {full_key: weights}
    with pytest.raises(ValueError, match="Duplicate"):
        dense._remap_lora_keys({key: weights, full_key: weights}, modules, packed, ["language_model"])
    with pytest.raises(ValueError, match="Ambiguous"):
        dense._remap_lora_keys({"proj": weights}, ["a.proj", "b.proj"], {}, ["a", "b"])
