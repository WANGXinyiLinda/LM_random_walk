from dataclasses import dataclass, field
from typing import Optional
import torch
import os
import sys

import transformers
from model.my_trainer import MyTrainer
transformers.logging.set_verbosity_info()
from huggingface_hub import login
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training, get_peft_model, PrefixTuningConfig, PromptTuningConfig, PromptTuningInit
from transformers.utils import logging

from model.load_model import MyAutoModelForCausalLM
from load_data.supervised_dataset import make_supervised_data_module
from load_data.random_walk_dataset import make_random_walk_data_module
from load_data.preprocess import GSMData, MathData, AquaData, SVAMPData
from load_data.build_graph import GSMGraph, MathGraph, AquaGraph, SVAMPGraph
from load_data.k_shot_dataset import KshotDataset
from model.peft_model import MyPeftModelForCausalLM
from model.utils import model_name_mapping

INVALID_ANS = "[invalid]"

logger = logging.get_logger(__name__)

@dataclass
class ModelArguments:
    random_initialize: Optional[bool] = field(default=False)
    model_name_or_path: Optional[str] = field(default="llama-2")
    base_model_name_or_path: Optional[str] = field(default="llama-2")
    parameter_efficient_mode: Optional['str'] = field(default='none', 
        metadata={"choices": ["none", "prompt-tuning", "lora", "prefix-tuning"]})
    hf_hub_token: Optional[str] = field(default=None, metadata={"help": "Require for llama family."})
    use_calculator: Optional[bool] = field(default=False)
    decoding_scheme: Optional[str] = field(default="greedy")
    cluster_model_name: Optional[str] = field(default="k-means", 
        metadata={"choices": ["vae", "k-means"]})
    num_nodes: Optional[int] = field(default=100)
    embedding_model_name: Optional[str] = field(default="llama-2")
    lora_module: Optional[str] = field(default="mlp")
    flash_attention: Optional[bool] = field(default=True)

@dataclass
class DataArguments:
    data_dir: str = field(default="data", metadata={"help": "directory to store processed datasets."}) 
    dataset: str = field(default="gsm8k", metadata={"help": "dataset name on huggingface."})
    mode: str = field(default="supervised", metadata={"choices": ["supervised", "random_walk"]})
    use_demonstrations: Optional[bool] = field(default=False)
    demo_selection: Optional[str] = field(default="uniform")
    candidate_size: Optional[int] = field(default=100)
    k_shot: Optional[int] = field(default=4)
    num_test: Optional[int] = field(default=2000)
    prompt_template: Optional[str] = field(default=None)
    random_seed: Optional[int] = field(default=42)
    cot_length: Optional[int] = field(default=10)
    continuous_steps: Optional[int] = field(default=5)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    resume: Optional[bool] = field(default=False)
    int8_training: Optional[bool] = field(default=False)
    load_in_16fp: Optional[bool] = field(default=False)
    continue_training: Optional[bool] = field(default=False)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def enable_prompt_tuning(model):
    model.get_input_embeddings().new_embedding.weight.requires_grad = True
    model.get_output_embeddings().new_linear.weight.requires_grad = True
    

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    login(token=model_args.hf_hub_token)

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    sys.stdout = open(f'{training_args.output_dir}/stdout.txt', 'w')

    if training_args.continue_training:
        model_name_or_path = model_name_mapping(model_args.base_model_name_or_path)
    else:
        model_name_or_path = model_name_mapping(model_args.model_name_or_path)

    if 'llama' in model_name_or_path or 'alpaca' in model_name_or_path:
        tokenizer = transformers.LlamaTokenizer.from_pretrained(
            model_name_or_path, cache_dir=training_args.cache_dir,
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path, cache_dir=training_args.cache_dir,
        )

    tokenizer.model_max_length = training_args.model_max_length
        
    special_tokens_dict = dict()
    if tokenizer.eos_token is None:
        if tokenizer.bos_token is None:
            special_tokens_dict["eos_token"] = '</s>'
        else:
            special_tokens_dict["eos_token"] = tokenizer.bos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    print("new token dict: ", special_tokens_dict)

    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)

    print("num new tokens: ", num_new_tokens)

    if data_args.mode == "supervised":

        if data_args.dataset == "gsm8k":
            data_class = GSMData
        elif data_args.dataset == "math":
            data_class = MathData
        elif data_args.dataset == "aqua":
            data_class = AquaData
        elif data_args.dataset == "svamp":
            data_class = SVAMPData
        else:
            raise NotImplementedError
        
        dataset = data_class("train", [], 
                            prompt_template=data_args.prompt_template,
                            tokenizer=tokenizer)

        if data_args.use_demonstrations:
            dataset = KshotDataset(dataset, dataset, data_args.k_shot,
                                data_args.demo_selection)

        eval_dataset = data_class("test", [], 
                            prompt_template=data_args.prompt_template,
                            tokenizer=tokenizer)

        print("train data: ", dataset[0])

        data_module = make_supervised_data_module(tokenizer, dataset, eval_dataset, 
                    max_num_eval=data_args.num_test, seed=data_args.random_seed)

    elif data_args.mode == "random_walk":

        if data_args.dataset == "gsm8k":
            data_class = GSMGraph
        elif data_args.dataset == "math":
            data_class = MathGraph
        elif data_args.dataset == "aqua":
            data_class = AquaGraph
        elif data_args.dataset == "svamp":
            data_class = SVAMPGraph
        else:
            raise NotImplementedError
        
        dataset = data_class("train", 
                    cluster_model_name=model_args.cluster_model_name, 
                    embedding_model_name=model_args.embedding_model_name,
                    num_nodes=model_args.num_nodes, 
                    data_dir=data_args.data_dir, 
                    tokenizer=tokenizer, 
                    prompt_template=data_args.prompt_template, 
                    cache_dir=training_args.cache_dir  )

        data_module = make_random_walk_data_module(tokenizer, dataset, 
                    seq_length=training_args.model_max_length,
                    cot_length=data_args.cot_length, 
                    continuous_steps=data_args.continuous_steps,
                    max_num_eval=data_args.num_test, 
                    seed=data_args.random_seed)
    
    else:
        print("no such data mode: ", data_args.mode)
        exit(1)

    print("Using pre-trained model weights...")

    if training_args.load_in_16fp or training_args.int8_training:
        model = MyAutoModelForCausalLM.from_pretrained(n_tokens=num_new_tokens,
            parameter_efficient_mode=model_args.parameter_efficient_mode,
            pretrained_model_name_or_path=model_name_or_path,
            cache_dir=training_args.cache_dir, torch_dtype=torch.float16, 
            device_map="auto", load_in_8bit=training_args.int8_training,
            offload_folder="offload", offload_state_dict = True,
            flash_attention=model_args.flash_attention,
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    else:
        model = MyAutoModelForCausalLM.from_pretrained(n_tokens=num_new_tokens,
            parameter_efficient_mode=model_args.parameter_efficient_mode,
            pretrained_model_name_or_path=model_name_or_path,
            cache_dir=training_args.cache_dir, 
            flash_attention=model_args.flash_attention,
        )

    if 'lora' in model_args.parameter_efficient_mode:

        if training_args.continue_training:
            model = MyPeftModelForCausalLM.from_pretrained(model, 
                model_args.model_name_or_path, is_trainable=True)
            
        else:
            target_modules = []
            if 'llama' in model_args.model_name_or_path \
                or 'alpaca' in model_args.model_name_or_path \
                or 'mistral' in model_args.model_name_or_path \
                or 'llemma' in model_args.model_name_or_path:
                if model_args.lora_module == 'mlp':
                    target_modules += ["gate_proj", "up_proj", "down_proj"]
                if model_args.lora_module == 'atten':
                    target_modules += ["q_proj", "k_proj", "v_proj", "o_proj"]
            elif 'gpt2' in model_args.model_name_or_path:
                target_modules = ["c_attn", "c_proj"]
            elif 'phi-2' in model_args.model_name_or_path:
                if model_args.lora_module == 'mlp':
                    target_modules += ["fc1", "fc2"]
                if model_args.lora_module == 'atten':
                    target_modules += ["Wqkv", "out_proj"]
            else:
                raise NotImplementedError
            
            peft_config = LoraConfig(r=16, lora_alpha=16, target_modules=target_modules, 
                                    lora_dropout=0.05, bias="none", inference_mode=False,
                                    task_type=TaskType.CAUSAL_LM)
            model = MyPeftModelForCausalLM(model, peft_config)

        if "prompt-tuning" in model_args.parameter_efficient_mode:
            enable_prompt_tuning(model.base_model.model)

    elif 'prefix-tuning' in model_args.parameter_efficient_mode:
        peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, 
                                         num_virtual_tokens=model_args.num_general_prefix_tokens)
        model = get_peft_model(model, peft_config)

    elif 'prompt-tuning' in model_args.parameter_efficient_mode:
        if model_args.only_at_front:
            peft_config = PromptTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                prompt_tuning_init=PromptTuningInit.TEXT,
                num_virtual_tokens=model_args.num_general_prefix_tokens,
                prompt_tuning_init_text="Solve the following math problem step-by-step:",
                tokenizer_name_or_path=model_name_or_path,
            )
            model = get_peft_model(model, peft_config)
        else:
            for p in model.parameters():
                p.requires_grad = False
            enable_prompt_tuning(model)
    
    print_trainable_parameters(model)
    # exit(1)

    tokenizer.padding_side = "left"
    trainer = MyTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    if training_args.resume:
        trainer.train(model_args.model_name_or_path)
    else:
        trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()