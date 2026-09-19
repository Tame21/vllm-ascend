# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from vllm_ascend.patch.worker import patch_lora_acl_graph as lora_graph


def _model_config(
    *,
    text_model_type,
    hf_model_type,
    architectures=(),
    multimodal=False,
    model="model",
    with_lora=True,
):
    return SimpleNamespace(
        lora_config=object() if with_lora else None,
        model_config=SimpleNamespace(
            model=model,
            hf_text_config=SimpleNamespace(model_type=text_model_type),
            hf_config=SimpleNamespace(
                model_type=hf_model_type,
                architectures=list(architectures),
            ),
            multimodal_config=object() if multimodal else None,
        ),
    )


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (
            _model_config(
                text_model_type="qwen3_5_text",
                hf_model_type="qwen3_5_text",
            ),
            True,
        ),
        (
            _model_config(
                text_model_type="mistral",
                hf_model_type="mistral3",
                architectures=("Mistral3ForConditionalGeneration",),
                multimodal=True,
                model="mistralai/Magistral-Small-2509",
            ),
            True,
        ),
        (
            _model_config(
                text_model_type="mistral",
                hf_model_type="mistral",
                architectures=("MistralForCausalLM",),
                multimodal=True,
                model="mistralai/Magistral-Small-2509",
            ),
            True,
        ),
        (
            _model_config(
                text_model_type="mistral",
                hf_model_type="mistral",
                architectures=("MistralForCausalLM",),
            ),
            False,
        ),
        (
            _model_config(
                text_model_type="mistral",
                hf_model_type="mistral3",
                multimodal=True,
                with_lora=False,
            ),
            False,
        ),
    ],
)
def test_patch_applies_to_qwen_and_magistral(config, expected):
    assert lora_graph.patch_applies(config) is expected


def test_lora_manager_marks_each_punica_wrapper_for_graph_routing():
    shared_wrapper = SimpleNamespace()
    language_wrapper = SimpleNamespace()
    manager = SimpleNamespace()

    def initialize_manager(*args, **kwargs):
        manager.punica_wrapper_mapping = {
            "language_model": language_wrapper,
            "vision_tower": shared_wrapper,
            "multi_modal_projector": shared_wrapper,
        }

    with (
        patch.object(lora_graph, "validate_config") as validate,
        patch.object(lora_graph, "specialize_lora", return_value=True),
        patch.object(
            lora_graph,
            "_ORIGINAL_MANAGER_INIT",
            side_effect=initialize_manager,
        ) as original,
    ):
        lora_graph._lora_manager_init(
            manager,
            "model",
            8,
            64,
            32000,
            "lora_config",
            "npu",
            "vllm_config",
        )

    validate.assert_called_once_with("vllm_config")
    original.assert_called_once()
    assert language_wrapper._ascend_graph_specialize_lora is True
    assert shared_wrapper._ascend_graph_specialize_lora is True


@pytest.mark.parametrize(
    ("has_lora", "num_tokens", "expected"),
    [
        (True, 1, lora_graph._AOT_LORA),
        (False, 1, lora_graph._AOT_BASE_ONE),
        (False, 2, lora_graph._AOT_BASE),
    ],
)
def test_select_variant(has_lora, num_tokens, expected):
    model = SimpleNamespace(_ascend_has_lora=lambda: has_lora)
    context = SimpleNamespace(batch_descriptor=SimpleNamespace(num_tokens=num_tokens))
    with (
        patch.object(lora_graph, "is_forward_context_available", return_value=True),
        patch.object(lora_graph, "get_forward_context", return_value=context),
    ):
        assert lora_graph._select_variant(model) == expected


def test_update_graph_params_filters_gdn_metadata():
    attention_metadata = SimpleNamespace(seq_lens_list=[4], actual_seq_lengths_q=[4])
    gdn_metadata = SimpleNamespace(num_prefills=1)
    forward_context = SimpleNamespace(
        attn_metadata={
            "model.layers.0.self_attn.attn": attention_metadata,
            "model.layers.1.linear_attn.attn": gdn_metadata,
        }
    )
    config = SimpleNamespace(
        lora_config=object(),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="qwen3_5_text")),
    )

    with (
        patch.object(
            lora_graph,
            "_EXTRA_CTX",
            SimpleNamespace(is_draft_model=False),
        ),
        patch.object(lora_graph, "using_paged_attention", return_value=False),
        patch.object(lora_graph, "_ORIGINAL_UPDATE_GRAPH_PARAMS") as original,
    ):
        lora_graph.update_graph_params("stream", forward_context, 4, config)

    # Filtering must not mutate metadata used by the GDN layers themselves.
    assert len(forward_context.attn_metadata) == 2
    filtered_context = original.call_args.args[1]
    assert filtered_context is not forward_context
    assert filtered_context.attn_metadata == {"model.layers.0.self_attn.attn": attention_metadata}


@pytest.mark.parametrize(
    ("mapping", "expected_no_lora"),
    [([0, 0], True), ([0, 2], False)],
)
def test_update_lora_metadata_tracks_decode_lora_state(mapping, expected_no_lora):
    wrapper = SimpleNamespace(
        _ascend_graph_specialize_lora=True,
        no_lora=not expected_no_lora,
    )
    lora_mapping = SimpleNamespace(index_mapping=mapping)
    with patch.object(lora_graph, "_ORIGINAL_UPDATE_LORA_METADATA") as original:
        lora_graph._update_lora_metadata(wrapper, lora_mapping)

    original.assert_called_once_with(wrapper, lora_mapping)
    assert wrapper.no_lora is expected_no_lora


def test_no_lora_graph_guard_skips_lora_operator():
    original = MagicMock(return_value="applied")
    guarded = lora_graph._no_lora_graph_guard(original)
    wrapper = SimpleNamespace(_ascend_graph_specialize_lora=True, no_lora=True)

    assert guarded(wrapper, "input") is None
    original.assert_not_called()


def test_aot_dispatcher_routes_loaded_variants():
    model = object()
    dispatcher = lora_graph._AOTVariantDispatcher()
    artifacts = {variant: MagicMock(return_value=variant) for variant in lora_graph._AOT_VARIANTS}
    for variant, artifact in artifacts.items():
        dispatcher.add_loaded(variant, artifact)

    for variant in lora_graph._AOT_VARIANTS:
        with patch.object(lora_graph, "_select_variant", return_value=variant):
            assert dispatcher(model, "input") == variant
        artifacts[variant].assert_called_once_with(model, "input")


def test_aot_dispatcher_lazily_compiles_missing_variant(monkeypatch):
    artifact = MagicMock(return_value="output")
    base_callable = MagicMock()
    base_callable.aot_compile.return_value = artifact
    model = SimpleNamespace(
        _compiled_callable=MagicMock(),
        _ascend_base_callable=base_callable,
        _ascend_base_one_callable=MagicMock(),
        _ascend_mark_variant_dynamic_inputs=MagicMock(),
        vllm_config=MagicMock(),
        _is_encoder=False,
    )
    dispatcher = lora_graph._AOTVariantDispatcher()
    monkeypatch.setattr(lora_graph.vllm_envs, "VLLM_DISABLE_COMPILE_CACHE", True)
    monkeypatch.setattr(lora_graph.compilation_counter, "num_aot_compiles", 0)

    with (
        patch.object(lora_graph, "_select_variant", return_value=lora_graph._AOT_BASE),
        patch.object(
            lora_graph.monitor,
            "monitor_torch_compile",
            return_value=nullcontext(),
        ),
    ):
        assert dispatcher(model, "input") == "output"

    model._ascend_mark_variant_dynamic_inputs.assert_called_once_with(lora_graph._AOT_BASE, "input")
    base_callable.aot_compile.assert_called_once_with((("input",), {}))
    artifact.assert_called_once_with(model, "input")


def test_try_loads_each_aot_variant(tmp_path):
    model = SimpleNamespace(_ascend_specialize_lora=True)
    artifacts = {variant: MagicMock(name=variant) for variant in lora_graph._AOT_VARIANTS}
    aot_path = str(tmp_path / "model")

    def load_artifact(_model, path):
        return artifacts[path.rsplit(".", 1)[-1]]

    with (
        patch.object(
            lora_graph,
            "_ORIGINAL_TRY_LOAD_AOT_COMPILED_FN",
            side_effect=load_artifact,
        ) as load,
        patch.object(lora_graph, "_select_variant", return_value=lora_graph._AOT_LORA),
    ):
        dispatcher = lora_graph._try_load_aot_compiled_fn(model, aot_path)

    assert isinstance(dispatcher, lora_graph._AOTVariantDispatcher)
    assert dispatcher.artifacts == artifacts
    assert load.call_args_list == [call(model, f"{aot_path}.{variant}") for variant in lora_graph._AOT_VARIANTS]


def test_partial_aot_cache_is_reused_when_selected_variant_is_missing(tmp_path):
    base_artifact = MagicMock(name="base")
    lora_artifact = MagicMock(name="lora")
    lora_callable = MagicMock()
    lora_callable.aot_compile.return_value = lora_artifact
    model = SimpleNamespace(
        _ascend_specialize_lora=True,
        _compiled_callable=lora_callable,
        _ascend_base_callable=MagicMock(),
        _ascend_base_one_callable=MagicMock(),
        _ascend_mark_variant_dynamic_inputs=MagicMock(),
    )
    aot_path = str(tmp_path / "model")

    def load_artifact(_model, path):
        return base_artifact if path.endswith(f".{lora_graph._AOT_BASE}") else None

    with (
        patch.object(
            lora_graph,
            "_ORIGINAL_TRY_LOAD_AOT_COMPILED_FN",
            side_effect=load_artifact,
        ),
        patch.object(lora_graph, "_select_variant", return_value=lora_graph._AOT_LORA),
    ):
        assert lora_graph._try_load_aot_compiled_fn(model, aot_path) is None
        dispatcher = lora_graph._aot_compile(model, "input")

    assert dispatcher.artifacts == {
        lora_graph._AOT_BASE: base_artifact,
        lora_graph._AOT_LORA: lora_artifact,
    }
    assert not hasattr(model, "_ascend_preloaded_aot_dispatcher")


def test_saves_each_aot_variant_to_an_independent_path(tmp_path, monkeypatch):
    class Artifact:
        def __init__(self, value):
            self.value = value

        def save_compiled_function(self, path):
            Path(path).write_text(self.value, encoding="utf-8")

    dispatcher = lora_graph._AOTVariantDispatcher()
    dispatcher.artifacts = {variant: Artifact(variant) for variant in lora_graph._AOT_VARIANTS}
    dispatcher.dirty_variants = set(lora_graph._AOT_VARIANTS)
    model = SimpleNamespace(
        _aot_cache_dir=str(tmp_path),
        _aot_compilation_path=str(tmp_path / "model"),
    )
    monkeypatch.setattr(lora_graph.vllm_envs, "VLLM_DISABLE_COMPILE_CACHE", False)
    monkeypatch.setattr(lora_graph.compilation_counter, "num_aot_artifacts_saved", 0)

    dispatcher.save_dirty(model)

    assert not dispatcher.dirty_variants
    for variant in lora_graph._AOT_VARIANTS:
        path = tmp_path / f"model.{variant}"
        assert path.read_text(encoding="utf-8") == variant
