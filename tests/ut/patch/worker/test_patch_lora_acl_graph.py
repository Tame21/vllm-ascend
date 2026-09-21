# SPDX-License-Identifier: Apache-2.0

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
        (True, 1, lora_graph._VARIANT_LORA),
        (False, 1, lora_graph._VARIANT_BASE_ONE),
        (False, 2, lora_graph._VARIANT_BASE),
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


@pytest.mark.parametrize(
    ("aot_enabled", "extension_enabled", "extension_installed", "specialize", "is_encoder", "reject"),
    [
        (True, True, False, True, False, True),
        (True, True, True, True, False, False),
        (True, False, False, True, False, False),
        (False, True, False, True, False, False),
        (False, False, False, True, False, False),
        (True, True, False, False, False, False),
        (True, True, False, True, True, False),
    ],
)
def test_wrapper_init_requires_aot_extension_only_for_specialized_aot(
    monkeypatch, aot_enabled, extension_enabled, extension_installed, specialize, is_encoder, reject
):
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            mode=lora_graph.CompilationMode.VLLM_COMPILE,
            dynamic_shapes_config=SimpleNamespace(evaluate_guards=False),
        )
    )
    native_aot_compile = MagicMock()
    native_aot_saver = MagicMock()
    model = SimpleNamespace(aot_compile=native_aot_compile, save_aot_compiled_function=native_aot_saver)
    monkeypatch.setattr(lora_graph.vllm_envs, "VLLM_USE_AOT_COMPILE", aot_enabled)
    monkeypatch.setattr(lora_graph.ascend_envs, "VLLM_ASCEND_ENABLE_LORA_AOT", extension_enabled)
    monkeypatch.setattr(lora_graph.vllm_envs, "VLLM_USE_BYTECODE_HOOK", True)
    monkeypatch.setattr(lora_graph.backends, "model_tag", "backbone")
    monkeypatch.setattr(
        lora_graph.TorchCompileWithNoGuardsWrapper,
        "_ascend_lora_aot_patch_installed",
        extension_installed,
        raising=False,
    )
    with (
        patch.object(lora_graph, "get_current_vllm_config", return_value=config),
        patch.object(lora_graph, "specialize_lora", return_value=specialize),
        patch.object(lora_graph, "validate_config"),
        patch.object(lora_graph, "_ORIGINAL_WRAPPER_INIT") as original,
        patch.object(lora_graph, "_compile_base_variant") as compile_base,
        patch.object(lora_graph.logger, "warning_once") as warning,
    ):
        if reject:
            with pytest.raises(ValueError, match="requires patch_lora_aot"):
                lora_graph._wrapper_init(model, "model", is_encoder=is_encoder)
            original.assert_not_called()
            compile_base.assert_not_called()
        else:
            lora_graph._wrapper_init(model, "model", is_encoder=is_encoder)
            original.assert_called_once_with(model, compile_prefix="model", is_encoder=is_encoder)
            assert model._ascend_specialize_lora is (specialize and not is_encoder)
            assert compile_base.call_count == (2 if model._ascend_specialize_lora else 0)
        assert lora_graph.vllm_envs.VLLM_USE_BYTECODE_HOOK is True
        assert model.aot_compile is native_aot_compile
        assert model.save_aot_compiled_function is native_aot_saver
        assert warning.call_count == int(aot_enabled and not extension_enabled and specialize and not is_encoder)


@pytest.mark.parametrize(("value", "expected"), [(None, True), ("1", True), ("0", False)])
def test_lora_aot_switch_defaults_to_enabled(monkeypatch, value, expected):
    name = "VLLM_ASCEND_ENABLE_LORA_AOT"
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)
    assert lora_graph.ascend_envs.env_variables[name]() is expected


@pytest.mark.parametrize("extension_enabled", [False, True])
def test_lora_aot_switch_does_not_disable_acl_graph_parameter_isolation(monkeypatch, extension_enabled):
    config = _model_config(text_model_type="qwen3_5_text", hf_model_type="qwen3_5_text")
    config.model_config.enforce_eager = False
    config.compilation_config = SimpleNamespace(
        cudagraph_mode=lora_graph.CUDAGraphMode.FULL,
        cudagraph_specialize_lora=True,
    )
    monkeypatch.setattr(lora_graph.ascend_envs, "VLLM_ASCEND_ENABLE_LORA_AOT", extension_enabled)
    monkeypatch.setattr(lora_graph.vllm_envs, "VLLM_USE_AOT_COMPILE", True)
    monkeypatch.setattr(lora_graph, "get_current_vllm_config", lambda: config)

    def initialize(params):
        params.events = {}
        params.workspaces = {}
        params.handles = {}
        params.attn_params = {}

    monkeypatch.setattr(lora_graph._ORIGINAL_GRAPH_PARAMS, "__init__", initialize)
    params = lora_graph._LoRAGraphParams()
    for name in ("events", "workspaces", "handles", "attn_params"):
        assert isinstance(getattr(params, name), lora_graph._GraphParamStore)


@pytest.mark.parametrize("extension_enabled", [False, True])
@pytest.mark.parametrize("release_matches", [False, True])
@pytest.mark.parametrize("is_310p", [False, True])
def test_worker_loader_preserves_graph_patch_when_aot_extension_is_disabled(
    extension_enabled, release_matches, is_310p
):
    # Execute the real release-lane import block without importing unrelated
    # worker patches. This checks both the switch and graph-before-AOT order.
    loader = Path(lora_graph.__file__).with_name("__init__.py")
    tree = ast.parse(loader.read_text(encoding="utf-8"))
    graph_module = "vllm_ascend.patch.worker.patch_lora_acl_graph"
    block = next(
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Import) and any(alias.name == graph_module for alias in child.names)
            for child in ast.walk(node)
        )
    )
    imported = []

    def record_import(name, *args, **kwargs):
        imported.append(name)
        return SimpleNamespace()

    namespace = {
        "__builtins__": {"__import__": record_import},
        "ascend_envs": SimpleNamespace(VLLM_ASCEND_ENABLE_LORA_AOT=extension_enabled),
        "vllm_version_is": lambda version: release_matches,
        "is_310p": lambda: is_310p,
    }
    exec(compile(ast.Module(body=[block], type_ignores=[]), str(loader), "exec"), namespace)
    expected = []
    if release_matches and not is_310p:
        expected = ["vllm_ascend.patch.worker.patch_qwen3_5_dense_lora", graph_module]
        if extension_enabled:
            expected.append("vllm_ascend.patch.worker.patch_lora_aot")
    assert imported == expected


@pytest.mark.parametrize(
    ("has_lora", "num_tokens", "expected"),
    [(True, 2, "lora"), (False, 2, "base"), (False, 1, "base_one")],
)
def test_ordinary_compile_routes_variants_without_aot_dispatcher(has_lora, num_tokens, expected):
    model = SimpleNamespace(
        _ascend_specialize_lora=True,
        _ascend_has_lora=lambda: has_lora,
        _compiled_callable=MagicMock(return_value="lora"),
        _ascend_base_callable=MagicMock(return_value="base"),
        _ascend_base_one_callable=MagicMock(return_value="base_one"),
        _ascend_base_dynamic_inputs_marked=False,
        _ascend_mark_variant_dynamic_inputs=MagicMock(),
        first_compile=True,
        evaluate_guards=False,
        _call_with_optional_nvtx_range=lambda fn, *args, **kwargs: fn(*args, **kwargs),
    )
    context = SimpleNamespace(batch_descriptor=SimpleNamespace(num_tokens=num_tokens))
    with (
        patch.object(lora_graph, "is_forward_context_available", return_value=True),
        patch.object(lora_graph, "get_forward_context", return_value=context),
        patch.object(lora_graph.wrapper_module, "_compilation_context", side_effect=nullcontext),
    ):
        assert lora_graph._wrapper_call(model, "input") == expected
        assert lora_graph._wrapper_call(model, "input") == expected
    assert model.first_compile is False
    assert model._ascend_mark_variant_dynamic_inputs.call_count == (1 if expected == "base" else 0)
