from dataclasses import dataclass, field
from tqdm import tqdm
from typing import Dict, Optional, Sequence
import transformers
import sentence_transformers
import torch
from torch.utils.data import DataLoader
import os, json, random, pickle
import numpy as np
from huggingface_hub import login

from load_data.preprocess import GSMData, MathData, AquaData, SVAMPData
from load_data.k_shot_dataset import KshotDataset
import calculator
from model.generation_utils import make_sparse_mask
from model.load_model import MyAutoModelForCausalLM
from model.peft_model import MyPeftModelForCausalLM
from model.utils import model_name_mapping

INVALID_ANS = "[invalid]"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="gpt2")
    base_model_name_or_path: Optional[str] = field(default="gpt2")
    cache_dir: Optional[str] = field(default=None)
    output_dir: Optional[str] = field(default=None, metadata={"help": "Path to the output dir."})
    max_length: Optional[int] = field(default=512)
    decoding_scheme: Optional[str] = field(default="greedy")
    load_in_8bit: Optional[bool] = field(default=False)
    use_calculator: Optional[bool] = field(default=False)
    parameter_efficient_mode: Optional['str'] = field(default='none', metadata={"choices": ["none", "prompt-tuning", "lora", "lora+prompt-tuning"]})
    hf_hub_token: Optional[str] = field(default=None, metadata={"help": "Require for llama family."})
    enable_cpu_offload: Optional[bool] = field(default=False)
    flash_attention: Optional[bool] = field(default=True)

@dataclass
class DataArguments:
    dataset: str = field(default=None, metadata={"help": "dataset name."})
    batch_size: Optional[int] = field(default=16)
    use_demonstrations: Optional[bool] = field(default=False)
    demo_selection: Optional[str] = field(default="uniform")
    candidate_size: Optional[int] = field(default=100)
    k_shot: Optional[int] = field(default=4)
    seed: Optional[int] = field(default=42)
    num_test: Optional[int] = field(default=1000)
    prompt_template: Optional[str] = field(default=None)
    embedding_model_name: Optional[str] = field(default='all-mpnet-base-v2')


def main():

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    login(token=model_args.hf_hub_token)
    random.seed(data_args.seed)

    if model_args.output_dir is None:
        model_args.output_dir = model_args.model_name_or_path
    
    os.makedirs(model_args.output_dir, exist_ok = True)

    if 'llama' in model_args.model_name_or_path or 'alpaca' in model_args.model_name_or_path:
        tokenizer = transformers.LlamaTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )
    print("loaded tokenizer")

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    if model_args.parameter_efficient_mode != 'none':
        model_name = model_name_mapping(model_args.base_model_name_or_path)
    else:
        model_name = model_args.model_name_or_path

    if 'prompt-tuning' in model_args.parameter_efficient_mode:
        input_embedding_file = model_args.model_name_or_path + '/embeddings.pt'
        output_embedding_file = None
        if not os.path.exists(input_embedding_file):
            input_embedding_file = model_args.model_name_or_path + '/input_embeddings.pt'
            output_embedding_file = model_args.model_name_or_path + '/output_embeddings.pt'
    else:
        input_embedding_file = None
        output_embedding_file = None

    if model_args.load_in_8bit:
        quantization_config = transformers.BitsAndBytesConfig(
            llm_int8_enable_fp32_cpu_offload=model_args.enable_cpu_offload)
        model = MyAutoModelForCausalLM.from_pretrained(
            input_embedding_file=input_embedding_file,
            output_embedding_file=output_embedding_file,
            pretrained_model_name_or_path=model_name,
            parameter_efficient_mode=model_args.parameter_efficient_mode,
            cache_dir=model_args.cache_dir, torch_dtype=torch.float16, 
            device_map="cuda", load_in_8bit=True,
            offload_folder="offload", offload_state_dict = True,
            quantization_config=quantization_config,
            flash_attention=model_args.flash_attention,
        )
    else:
        model = MyAutoModelForCausalLM.from_pretrained(
            input_embedding_file=input_embedding_file,
            output_embedding_file=output_embedding_file,
            pretrained_model_name_or_path=model_name,
            parameter_efficient_mode=model_args.parameter_efficient_mode,
            cache_dir=model_args.cache_dir,
            device_map="auto", torch_dtype=torch.float32,  
            offload_folder="offload", offload_state_dict = True,
            flash_attention=model_args.flash_attention,
        )
        
    
    if 'lora' in model_args.parameter_efficient_mode:
        model = MyPeftModelForCausalLM.from_pretrained(model, 
            model_args.model_name_or_path)
            
    print("loaded model.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()  
    
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

    dataset = data_class("test", [], 
                        prompt_template=data_args.prompt_template,
                        tokenizer=tokenizer,)

    if len(dataset) > data_args.num_test:
        idx = random.choices(list(range(len(dataset))), k=data_args.num_test)
        new_x = []
        new_y = []
        for i in idx:
            new_x.append(dataset[i]['x'])
            new_y.append(dataset[i]['y'])
        dataset.x = new_x
        dataset.y = new_y

    assert len(dataset) <= data_args.num_test
    print(dataset[0], len(dataset))

    if data_args.use_demonstrations:

        demo_dataset = data_class("train", [], 
                        prompt_template=data_args.prompt_template,
                        tokenizer=tokenizer,)
        rand_ids = random.sample(range(len(demo_dataset)), data_args.candidate_size)
        demo_dataset = [demo_dataset[i] for i in rand_ids]
        save_dir = f'demos/{data_args.dataset}/gpt2-xl' #Llama-2-70b-hf

        if os.path.exists(save_dir + '/sorted_demo_data.json') or data_args.demo_selection != 'prompt':
            dataset = KshotDataset(dataset, demo_dataset, data_args.k_shot,
                                data_args.demo_selection, save_dir=save_dir)
        else:
            dataset = KshotDataset(dataset, demo_dataset, data_args.k_shot,
                                    data_args.demo_selection, model, tokenizer, 
                                    None, save_dir)
            print("selected demos: ", dataset[0]['x'])
            print("prompt losses calculated")
            exit(0)

    class KeywordsStoppingCriteria(transformers.StoppingCriteria):
        def __init__(self, keywords_ids:list):
            self.keywords = keywords_ids

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            stop = True
            for i, k in enumerate(self.keywords):
                if input_ids[0][i-len(self.keywords)] != k:
                    stop = False
            return stop
        
    stop_ids = tokenizer.encode('500\n\n')[-2:]
    print(stop_ids)
    stop_criteria = KeywordsStoppingCriteria(stop_ids)

    print("loaded dataset")
    
    dataloader = DataLoader(dataset, batch_size=data_args.batch_size, shuffle=False)

    if data_args.use_demonstrations:
        out_file_name = f'{model_args.output_dir}/{data_args.dataset}_test_cal={model_args.use_calculator}_demo={data_args.demo_selection}_k={data_args.k_shot}_output.txt'
    else:
        out_file_name = f'{model_args.output_dir}/{data_args.dataset}_test_cal={model_args.use_calculator}_output.txt'
            
    output = []
    num_correct = 0
    num_all = 0

    for i, batch in tqdm(enumerate(dataloader)):
        x_text, y_text = batch['x'], batch['y']
        if model_args.use_calculator:
            generated_texts = []
            for text in x_text:
                generated_texts.append(calculator.sample(model, text, tokenizer, 
                    device, model_args.max_length, stop_ids))
        else:
            if data_args.use_demonstrations:
                generated_texts = []
                for x in x_text:
                    print(x)
                    encoding = tokenizer([x], padding=True, return_tensors='pt').to(device)
                    max_length = min(model_args.max_length, encoding['input_ids'].size(1) + 512)
                    with torch.no_grad():
                        generated_ids = model.generate(**encoding, max_length=max_length,
                            stopping_criteria=transformers.StoppingCriteriaList([stop_criteria]))
                    generated_text = tokenizer.decode(generated_ids[0, 
                        encoding['input_ids'].size(1):], skip_special_tokens=True)
                    print(generated_text)
                    generated_texts.append(generated_text)
                    
            else:
                encoding = tokenizer(x_text, padding=True, return_tensors='pt').to(device)
                max_length = min(model_args.max_length, encoding['input_ids'].size(1) + 512)
                with torch.no_grad():
                    generated_ids = model.generate(**encoding, 
                        max_length=max_length)
                try:
                    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                except:
                    print("cannot decode: ")
                    print(generated_ids)

        for text, x, y in zip(generated_texts, x_text, y_text):
            text, x, y = str(text), str(x), str(y)
            print(text)
            if dataset.is_correct(text, y):
                num_correct += 1
                print('correct')
            else:
                print('wrong')
            
            num_all += 1
            output.append((text, y))

        with open(out_file_name, 'w') as f:
            for x, y in output:
                f.write(x.encode('ascii', 'ignore').decode('ascii') + '\n' + 
                        y.encode('ascii', 'ignore').decode('ascii') + '\n\n')
            f.write(f"Accuracy: {num_correct/num_all}")
        
        print("Accuracy: ", num_correct/num_all)
    
    print("Accuracy: ", num_correct/num_all)
    print("num test: ", num_all)
    
    with open(out_file_name, 'w') as f:
        for x, y in output:
            f.write(x.encode('ascii', 'ignore').decode('ascii') + '\n' + 
                    y.encode('ascii', 'ignore').decode('ascii') + '\n\n')
        f.write(f"Accuracy: {num_correct/num_all}")


if __name__ == "__main__":
    main()