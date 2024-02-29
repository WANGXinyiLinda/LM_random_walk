# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
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

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import PreTrainedModel

from peft.peft_model import PeftModel, PeftConfig, PeftModelForCausalLM


class MyPeftModel(PeftModel):

    def __init__(self, model: PreTrainedModel, peft_config: PeftConfig, 
                 adapter_name: str = "default", add_tokens=False):
        super().__init__(model, peft_config, adapter_name)
        self.add_tokens = add_tokens

    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = False,
        selected_adapters: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        super().save_pretrained(save_directory, safe_serialization, 
                                selected_adapters, **kwargs)

        if self.add_tokens:
            torch.save(self.base_model.model.get_input_embeddings().new_embedding, 
                        os.path.join(save_directory, "input_embeddings.pt"))
            torch.save(self.base_model.model.get_output_embeddings().new_linear, 
                        os.path.join(save_directory, "output_embeddings.pt"))


class MyPeftModelForCausalLM(MyPeftModel, PeftModelForCausalLM):

    def __init__(self, model, peft_config: PeftConfig, adapter_name="default", add_tokens=False):
        super().__init__(model, peft_config, adapter_name)
        self.add_tokens = add_tokens