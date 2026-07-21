#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from functools import wraps

from vllm.lora import model_manager, utils


def _wrap_supported_lora_modules(original):
    @wraps(original)
    def patched(model):
        return [name for name in original(model) if "in_proj_ba" not in name]

    return patched


if not hasattr(utils, "_ascend_original_get_supported_lora_modules"):
    utils._ascend_original_get_supported_lora_modules = utils.get_supported_lora_modules
    utils.get_supported_lora_modules = _wrap_supported_lora_modules(utils.get_supported_lora_modules)

if not hasattr(model_manager, "_ascend_original_get_supported_lora_modules"):
    model_manager._ascend_original_get_supported_lora_modules = model_manager.get_supported_lora_modules
    model_manager.get_supported_lora_modules = _wrap_supported_lora_modules(model_manager.get_supported_lora_modules)
