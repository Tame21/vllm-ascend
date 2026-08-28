# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Sleep/wakeup cleanup for all six target/draft Base/LoRA graph stores."""

from vllm_ascend.compilation import acl_graph


def clear_all_attention_workspaces(cls) -> None:
    for params in acl_graph.iter_graph_params():
        cls.clear_attention_workspaces(params)


def reset_all_graph_params(cls) -> None:
    for params in acl_graph.iter_graph_params():
        cls.reset_graph_params(params)
    # Preserve the existing cleanup after the changed lines in lora_2.patch.
    for wrapper in list(acl_graph._acl_graph_wrappers):
        wrapper.concrete_aclgraph_entries.clear()
        wrapper.first_run_finished = False
