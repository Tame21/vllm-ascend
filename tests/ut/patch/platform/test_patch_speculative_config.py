# SPDX-License-Identifier: Apache-2.0

import pytest
from vllm.config.speculative import SpeculativeConfig

# Importing the module installs the compatibility patch under test.
import vllm_ascend.patch.platform.patch_speculative_config  # noqa: F401


@pytest.mark.parametrize(
    ("method", "parallel_drafting", "expected_slots"),
    [
        pytest.param("eagle3", False, 0, id="eagle3"),
        pytest.param("eagle3", True, 7, id="parallel-eagle"),
        pytest.param("dflash", True, 8, id="dflash"),
        pytest.param("dspark", True, 7, id="dspark"),
        pytest.param("mtp", False, 0, id="mtp"),
        pytest.param("ngram", False, 0, id="ngram"),
        pytest.param("draft_model", False, 1, id="draft-model"),
        pytest.param("draft_model", True, 8, id="parallel-draft-model"),
    ],
)
def test_max_num_new_slots_for_drafting(method, parallel_drafting, expected_slots):
    speculative_config = SpeculativeConfig(model="ngram", num_speculative_tokens=8)
    speculative_config.method = method
    speculative_config.parallel_drafting = parallel_drafting

    assert speculative_config.max_num_new_slots_for_drafting == expected_slots
