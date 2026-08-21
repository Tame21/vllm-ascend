from __future__ import annotations

import pytest
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.config import CompilationConfig
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.conftest import VllmRunner
from tests.e2e.pull_request.one_card.spec_decode.utils import BASELINES, DSPARK, calculate_acceptance_per_pos


@pytest.mark.parametrize("method", DSPARK.keys())
@pytest.mark.parametrize("num_speculative_tokens", [7])
def test_dspark_acceptance(
    method: str,
    num_speculative_tokens: int,
):
    main_model_name = DSPARK[method]["main"]
    spec_model_name = DSPARK[method]["spec"]

    tokenizer = AutoTokenizer.from_pretrained(
        main_model_name,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=0,
        ignore_eos=False,
        max_tokens=256,
    )

    prompts = [{"role": "user", "content": "Hello, your name is"}]
    prompts = [
        tokenizer.apply_chat_template(
            [prompt],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in prompts
    ]

    speculative_config = {
        "method": "dspark",
        "model": spec_model_name,
        "num_speculative_tokens": num_speculative_tokens,
    }

    compilation_config = CompilationConfig(cudagraph_mode="PIECEWISE", cudagraph_capture_sizes=[7, 8])

    with VllmRunner(
        main_model_name,
        max_model_len=4096,
        disable_log_stats=False,
        tensor_parallel_size=1,
        max_num_seqs=256,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.8,
        speculative_config=speculative_config,
        compilation_config=compilation_config,
        enable_prefix_caching=False,
    ) as llm:
        outputs = llm.model.generate(prompts, sampling_params)
        metrics = llm.model.get_metrics()

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        output_tokens = output.outputs[0].token_ids
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        print(f"Output tokens: {output_tokens}")

    acceptance_per_pos = calculate_acceptance_per_pos(metrics, num_speculative_tokens, Counter, Vector)
    golden = BASELINES[method]

    match = all(abs(a - b) < 0.1 for a, b in zip(acceptance_per_pos, golden))
    assert match, f"acceptance_per_pos {acceptance_per_pos} does not match golden {golden}"


def _dspark_prompts(tokenizer) -> list[str]:
    chat_prompts = [
        {"role": "user", "content": "Hello, your name is"},
        {"role": "user", "content": "The capital of France is"},
    ]
    return [
        tokenizer.apply_chat_template(
            [prompt],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in chat_prompts
    ]


def _run_dspark(main_model_name, spec_model_name, num_speculative_tokens, prompts, additional_config):
    speculative_config = {
        "method": "dspark",
        "model": spec_model_name,
        "num_speculative_tokens": num_speculative_tokens,
    }
    compilation_config = CompilationConfig(cudagraph_mode="PIECEWISE", cudagraph_capture_sizes=[7, 8])
    with VllmRunner(
        main_model_name,
        max_model_len=4096,
        disable_log_stats=False,
        tensor_parallel_size=1,
        max_num_seqs=256,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.8,
        speculative_config=speculative_config,
        compilation_config=compilation_config,
        enable_prefix_caching=False,
        additional_config=additional_config,
    ) as llm:
        sampling_params = SamplingParams(
            temperature=0,
            ignore_eos=False,
            max_tokens=256,
        )
        outputs = llm.model.generate(prompts, sampling_params)
        metrics = llm.model.get_metrics()
    return outputs, metrics


@pytest.mark.parametrize("method", DSPARK.keys())
@pytest.mark.parametrize("num_speculative_tokens", [7])
def test_dspark_decode_only_matches_prefill_tail(
    method: str,
    num_speculative_tokens: int,
):
    """DSpark decode_only must not change greedy outputs or acceptance.

    decode_only moves the DSpark context projection and the first proposal
    from the final prefill to after the first ordinary decode; after that
    the steady-state speculative loop is unchanged, so greedy token ids must
    match the prefill_tail baseline exactly and acceptance per position must
    stay within tolerance.
    """
    main_model_name = DSPARK[method]["main"]
    spec_model_name = DSPARK[method]["spec"]

    tokenizer = AutoTokenizer.from_pretrained(
        main_model_name,
        trust_remote_code=True,
    )
    prompts = _dspark_prompts(tokenizer)

    baseline_outputs, baseline_metrics = _run_dspark(
        main_model_name, spec_model_name, num_speculative_tokens, prompts, additional_config={}
    )

    decode_only_config = {
        "dspark_config": {
            "execution_phase": "decode_only",
            "staging_device": "npu",
            "max_staged_tokens_per_request": 4096,
            "max_staged_bytes_total": 2147483648,
            "lazy_init_chunk_tokens": 1024,
            "overflow_policy": "fallback_prefill_tail",
        }
    }
    decode_only_outputs, decode_only_metrics = _run_dspark(
        main_model_name,
        spec_model_name,
        num_speculative_tokens,
        prompts,
        additional_config=decode_only_config,
    )

    for baseline, decode_only in zip(baseline_outputs, decode_only_outputs):
        assert baseline.outputs[0].token_ids == decode_only.outputs[0].token_ids, (
            f"decode_only greedy output diverged from prefill_tail for prompt "
            f"{baseline.prompt!r}: {baseline.outputs[0].token_ids[:32]} vs "
            f"{decode_only.outputs[0].token_ids[:32]}"
        )

    baseline_acceptance = calculate_acceptance_per_pos(baseline_metrics, num_speculative_tokens, Counter, Vector)
    decode_only_acceptance = calculate_acceptance_per_pos(decode_only_metrics, num_speculative_tokens, Counter, Vector)
    assert all(abs(a - b) < 0.1 for a, b in zip(decode_only_acceptance, baseline_acceptance)), (
        f"decode_only acceptance {decode_only_acceptance} diverged from prefill_tail baseline {baseline_acceptance}"
    )
