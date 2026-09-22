# SPDX-License-Identifier: Apache-2.0
"""NPU regression for alternating base/adapter requests and AOT cache reload."""

import pytest
from vllm import SamplingParams
from vllm.lora.request import LoRARequest

from tests.e2e.conftest import VllmRunner


@pytest.mark.parametrize("use_aot", [False, True])
@pytest.mark.parametrize("rank", [8, 16])
def test_qwen35_lora_graph_routing_and_reload(qwen35_text_lora_files, monkeypatch, tmp_path, use_aot, rank):
    monkeypatch.setenv("VLLM_USE_AOT_COMPILE", str(int(use_aot)))
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_LORA_AOT", "1")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "0")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setenv("VLLM_FORCE_AOT_LOAD", "0")
    adapter_a = LoRARequest("adapter_a", 1, qwen35_text_lora_files)
    adapter_b = LoRARequest("adapter_b", 2, qwen35_text_lora_files)
    prompt = "Write a SQL query: How many singers are in the table singer(Singer_ID, Name, Age)?"
    sampling = SamplingParams(temperature=0, max_tokens=32, seed=0)
    first_outputs = None
    cache_times = None

    for restart in range(2 if use_aot else 1):
        if restart:
            # A failed load must fail this run instead of silently recompiling.
            monkeypatch.setenv("VLLM_FORCE_AOT_LOAD", "1")
        with VllmRunner(
            "Qwen/Qwen3.5-4B",
            enable_lora=True,
            max_lora_rank=rank,
            max_loras=2,
            max_num_seqs=4,
            max_num_batched_tokens=256,
            enable_prefix_caching=False,
            compilation_config={
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [1, 2, 4],
                "cudagraph_specialize_lora": True,
            },
        ) as runner:

            def generate(adapters):
                outputs = runner.model.generate(
                    [prompt] * len(adapters), sampling, lora_request=adapters, use_tqdm=False
                )
                return [output.outputs[0].token_ids for output in outputs]

            base = generate([None, None])
            adapted = generate([adapter_a, adapter_a])
            assert generate([None, None]) == base
            assert generate([adapter_b, adapter_b]) == adapted
            assert generate([None]) == [base[0]]
            assert generate([None, adapter_a, adapter_b]) == [base[0], adapted[0], adapted[0]]
            assert adapted != base, "The fixture adapter must affect this prompt"
            if restart:
                assert (base, adapted) == first_outputs
            else:
                first_outputs = (base, adapted)

        if use_aot:
            artifacts = [path for suffix in ("lora", "base", "base_one") for path in tmp_path.rglob(f"model.{suffix}")]
            assert {path.suffix for path in artifacts} == {".lora", ".base", ".base_one"}
            current_times = {str(path): path.stat().st_mtime_ns for path in artifacts}
            if restart:
                assert current_times == cache_times, "Restart must reuse the existing artifacts"
            cache_times = current_times
