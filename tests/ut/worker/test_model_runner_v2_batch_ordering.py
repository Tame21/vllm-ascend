# SPDX-License-Identifier: Apache-2.0

from vllm_ascend.worker.v2.model_runner import _sort_batch_req_ids


def test_verification_requests_stay_before_short_prefills():
    num_tokens_per_req = {
        "short_prefill": 2,
        "partial_verification": 4,
        "decode": 1,
        "full_verification": 9,
        "long_prefill": 32,
    }
    draft_tokens = {
        "partial_verification": [11, 12, 13],
        "full_verification": list(range(8)),
    }

    req_ids = _sort_batch_req_ids(
        num_tokens_per_req,
        draft_tokens,
        decode_query_len=9,
    )

    assert req_ids == [
        "full_verification",
        "partial_verification",
        "decode",
        "short_prefill",
        "long_prefill",
    ]


def test_empty_draft_list_is_not_a_verification_request():
    num_tokens_per_req = {
        "empty_draft": 2,
        "partial_verification": 3,
        "decode": 1,
    }
    draft_tokens = {
        "empty_draft": [],
        "partial_verification": [21, 22],
    }

    req_ids = _sort_batch_req_ids(
        num_tokens_per_req,
        draft_tokens,
        decode_query_len=9,
    )

    assert req_ids == ["partial_verification", "decode", "empty_draft"]
