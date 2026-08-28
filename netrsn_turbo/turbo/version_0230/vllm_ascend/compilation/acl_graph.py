# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Base/LoRA graph storage from lora_2.patch, installed by TurboManager."""

from collections.abc import Iterator
from typing import Any

import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.compilation import acl_graph

GraphParamsByLoRA = dict[bool, acl_graph.GraphParams]


def _new_graph_params(aclgraph_capture_sizes: list[int]) -> acl_graph.GraphParams:
    return acl_graph.GraphParams(
        {size: [] for size in aclgraph_capture_sizes},
        {size: None for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
    )


def _select_graph_params(
    params: acl_graph.GraphParams | GraphParamsByLoRA | None,
) -> acl_graph.GraphParams | None:
    if params is None or not isinstance(params, dict):
        return params
    try:
        forward_context = acl_graph.get_forward_context()
    except (AssertionError, LookupError, RuntimeError):
        return params[False]
    descriptor = forward_context.batch_descriptor
    has_lora = (
        forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
        and descriptor is not None
        and descriptor.has_lora
    )
    return params[has_lora]


def iter_graph_params() -> Iterator[acl_graph.GraphParams]:
    for params in (
        acl_graph._graph_params,
        acl_graph._draft_graph_params,
        acl_graph._draft_graph_prefill_params,
    ):
        if isinstance(params, dict):
            yield from params.values()
        elif params is not None:
            yield params


# State stays on the ORIGINAL module. A `global _graph_params` here would
# create a second store invisible to ACLGraphWrapper and sleep/wakeup.
def set_graph_params(aclgraph_capture_sizes: list[int]):
    if acl_graph._graph_params is not None:
        raise ValueError("Graph parameters have already been set!")
    acl_graph._graph_params = {
        has_lora: _new_graph_params(aclgraph_capture_sizes)
        for has_lora in (False, True)
    }


def get_graph_params():
    return _select_graph_params(acl_graph._graph_params)


def update_graph_params_workspaces(num_tokens: int, workspace: torch.Tensor):
    graph_params = get_graph_params()
    if graph_params is not None:
        graph_params.workspaces[num_tokens] = workspace


def set_draft_graph_params(aclgraph_capture_sizes: list[int]):
    if acl_graph._draft_graph_params is not None:
        raise ValueError("DraftGraph parameters have already been set!")
    acl_graph._draft_graph_params = {
        has_lora: _new_graph_params(aclgraph_capture_sizes)
        for has_lora in (False, True)
    }


def get_draft_graph_params():
    return _select_graph_params(acl_graph._draft_graph_params)


def update_draft_graph_params_workspaces(num_tokens: int, workspace: Any):
    graph_params = get_draft_graph_params()
    if graph_params is not None:
        graph_params.workspaces[num_tokens] = workspace


def set_draft_graph_prefill_params(aclgraph_capture_sizes: list[int]):
    if acl_graph._draft_graph_prefill_params is not None:
        raise ValueError("DraftGraph preill parameters have already been set!")
    acl_graph._draft_graph_prefill_params = {
        has_lora: _new_graph_params(aclgraph_capture_sizes)
        for has_lora in (False, True)
    }


def get_draft_graph_prefill_params():
    return _select_graph_params(acl_graph._draft_graph_prefill_params)


def update_draft_graph_prefill_params_workspaces(num_tokens: int, workspace: Any):
    graph_params = get_draft_graph_prefill_params()
    if graph_params is not None:
        graph_params.workspaces[num_tokens] = workspace


def weak_ref_workspaces(params):
    # The unmodified ACLGraphWrapper.__call__ passes the raw module stores.
    # Selecting here is equivalent to the three getter calls in lora_2.patch,
    # without copying the whole graph capture/replay implementation.
    params = _select_graph_params(params)
    if params is None:
        return
    for num_tokens in params.workspaces:
        if params.workspaces[num_tokens] is not None:
            params.workspaces[num_tokens] = acl_graph.weak_ref_tensors(
                params.workspaces[num_tokens]
            )
