from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import transformers
import torch
from torch.utils.data import Dataset, DataLoader
from string import Template
import os
import random
import numpy as np
from transformers import StoppingCriteria, StoppingCriteriaList

from data_generation.generator import Grapher

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="gpt2")
    cache_dir: Optional[str] = field(default=None)
    output_dir: Optional[str] = field(default=None, metadata={"help": "Path to the output dir."})
    max_length: Optional[int] = field(default=128)
    decoding_scheme: Optional[str] = field(default="greedy")
    max_num_reasoning_steps: Optional[int] = field(default=1)
    generate_n: Optional[int] = field(default=1)
    entity_as_new_token: Optional[bool] = field(default=True)
    relation_as_new_token: Optional[bool] = field(default=True)

@dataclass
class DataArguments:
    data_dir: str = field(default=None, metadata={"help": "Path to the training data."})
    dataset: str = field(default=None, metadata={"help": "dataset name."})
    batch_size: Optional[int] = field(default=32)
    split: Optional[str] = field(default="dev")
    use_demonstrations: Optional[bool] = field(default=False)
    mode: str = field(default="completion")
    randomize_entity_name: Optional[bool] = field(default=False)
    num_positive: Optional[int] = field(default=1)
    num_negative: Optional[int] = field(default=1)
    max_num_rules_per_r: Optional[int] = field(default=100)

class SupervisedDataset(Dataset):

    def __init__(self, data_dir: str, dataset: str, split: str, tokenizer=None, 
                mode='true_or_false', randomize=False, 
                entity_as_new_token=False, relation_as_new_token=False,
                max_rule_len=5, max_num_rules=100, 
                use_demonstrations=False, num_positive=1, num_negative=1, k=4, seed=1234):
        super(SupervisedDataset, self).__init__()
        self.data_dir = data_dir
        self.dataset = dataset
        self.split = split
        self.tokenizer = tokenizer
        self.eos_token = self.tokenizer.eos_token
        self.mode = mode
        self.use_demonstrations = use_demonstrations

        self.grapher = Grapher(data_dir, dataset, randomize, False,
                               entity_as_new_token, relation_as_new_token,
                               max_rule_len, max_num_rules, seed)
        if use_demonstrations:
            self.all_proofs = self.grapher.get_all_proofs()
        self.relation_as_new_token = relation_as_new_token
        
        if split == 'test':
            self.all_triples = self.grapher.test_triples
        elif split == 'dev':
            self.all_triples = self.grapher.dev_triples
        elif split == 'train':
            self.all_triples = self.grapher.train_triples
        else:
            print("no such split")
            exit(1)
        
        self.input_text = []
        self.label_text = []
        self.seen_answers = {}
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.k = k
        self.get_data()

    def parse_reasoning_path(self, generated_text, num_steps=-1, r=None):
        if self.mode == "true_or_false":
            generated_text = generated_text.split(' True or False? ')[-1]
        sents = generated_text.split('. ')
        if self.mode == "true_or_false":
            sents = sents[:-3]
        if num_steps > 0:
            sents = sents[:num_steps]
        proof = []
        for sent in sents:
            if num_steps == 1:
                triple = self.grapher.sent2triple(sent, r)
            else:
                triple = self.grapher.sent2triple(sent)
            if triple is None:
                proof.append(sent)
            else:
                proof.append(triple)
        return proof

    def verify(self, generated_text, y_text, goal_triple, num_steps=-1):
        if self.mode == "true_or_false":
            if generated_text[-len(y_text):] == y_text:
                return 'correct'
            else:
                return 'wrong'
        else:
            e1, r, e2s = goal_triple
            print("goal: ", goal_triple)
            print("seen answers: ", self.seen_answers[(e1, r)])
            proof = self.parse_reasoning_path(generated_text[:-1], num_steps, r)
            print("proof: ", proof)
            if self.mode == "completion":
                pred_e2 = proof[-1][-1]
                # print(pred_e2)
                if pred_e2 in e2s:
                    return 'correct'
                if 'countries' in self.dataset:
                    try:
                        for _r, _e in self.grapher.store[pred_e2]:
                            if _r == r and _e in e2s:
                                return 'correct'
                        return 'wrong'
                    except:
                        return 'cannot_parse'
                if pred_e2 in self.seen_answers[(e1, r)]:
                    return 'seen'
                if type(pred_e2) == int:
                    return 'wrong'
                else:
                    return 'cannot_parse'
            else:
                raise NotImplementedError


    def get_demonstrations(self, k=4):
        demos = []
        all_goal_triples = list(self.all_proofs.keys())
        goal_triples = random.sample(all_goal_triples, k=k)

        for goal in goal_triples:
            e1, r, e2 = goal
            rule = random.choice(list(self.all_proofs[goal].keys()))
            proof = random.choice(self.all_proofs[goal][rule])
            goal_sent = self.grapher.triple2sent(e1, r, e2)
            proof_sents = self.grapher.proof2sent(rule, proof)
            reasoning_chain = ' '.join(proof_sents)
            if self.mode == "true_or_false":
                demo = (goal_sent + ' True or False? ' + reasoning_chain
                            + ' So ' + goal_sent + ' True.')
            else:
                demo = reasoning_chain
            demos.append(demo)

        return demos

    def get_data(self):
        self.input_triples = []
        for e1, r, e2 in self.all_triples:
            if (e1, r) not in self.seen_answers:
                seen_answers = []
                for _r, _e in self.grapher.store[e1]:
                    if _r == r:
                        seen_answers.append(_e)
                self.seen_answers[(e1, r)] = seen_answers
            if e2 in self.seen_answers[(e1, r)]:
                print(self.grapher.triple2sent(e1, r, e2), 'is seen')

            if self.mode == "true_or_false":

                for i in range(self.num_positive):
                    goal_sent = self.grapher.triple2sent(e1, r, e2)
                    if self.use_demonstrations:
                        sents = self.get_demonstrations(k=self.k)
                        sents.append(goal_sent)
                        sent = self.eos_token.join(sents)
                    else:
                        sent = goal_sent
                    self.input_text.append(sent + ' True or False?')
                    self.label_text.append("True.")
                    self.input_triples.append((e1, r, e2))

                neg_e2, hard_neg, easy_neg = self.grapher.sample_negative_e2(e1, r, e2)

                if self.num_negative <= len(hard_neg):
                    negs = random.sample(hard_neg, k=self.num_negative)
                else:
                    negs = hard_neg
                    negs += random.sample(easy_neg, k=self.num_negative-len(hard_neg))
                
                for i in range(self.num_negative):
                    goal_sent = self.grapher.triple2sent(e1, r, e2)
                    neg_e2 = negs[i]
                    neg_sent = self.grapher.triple2sent(e1, r, neg_e2)
                    if self.use_demonstrations:
                        sents = self.get_demonstrations(k=self.k)
                        sents.append(neg_sent)
                        sent = self.eos_token.join(sents)
                    else:
                        sent = neg_sent
                    self.input_text.append(sent + ' True or False?')
                    self.label_text.append("False.")
                    self.input_triples.append((e1, r, e2))

            elif self.mode == "completion":
                prompt_exist = False
                answer = self.grapher.entity2name(e2)
                for triple, label in zip(self.input_triples, self.label_text):
                    if e1 == triple[0] and r == triple[1]:
                        triple[2].append(e2)
                        label.append(answer)
                        prompt_exist = True
                
                if not prompt_exist:
                    if self.relation_as_new_token:
                        query = ' '.join([self.grapher.entity2name(e1), f'<relation_{r}>'])
                    else:
                        temp = Template(self.grapher.relation2sent[r][0])
                        query = temp.substitute(e1=self.grapher.entity2name(e1), e2='')[:-2]
                    
                    if self.use_demonstrations:
                        sents = self.get_demonstrations(k=self.k)
                        sents.append(query)
                        sent = self.eos_token.join(sents)
                    else:
                        sent = query

                    self.input_text.append(sent)
                    self.label_text.append([answer])
                    self.input_triples.append([e1, r, [e2]])


    def __len__(self):
        return len(self.input_text)

    def __getitem__(self, i):
        return self.input_text[i], self.label_text[i], self.input_triples[i]

    
class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
      super().__init__()
      self.stops = stops
      self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
      stop_count = 0
      for stop in self.stops:
        stop_count = (stop == input_ids).sum(-1).min().item()

      if stop_count >= self.ENCOUNTERS:
          return True
      return False

def eval():

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_args.output_dir is None:
        model_args.output_dir = model_args.model_name_or_path
    else:
        os.makedirs(model_args.output_dir, exist_ok = True)

    # try:
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    ).to(device)
    # except:
    #     model = transformers.AutoModelForCausalLM.from_pretrained(
    #         model_args.model_name_or_path,
    #         cache_dir=model_args.cache_dir,
    #         use_safetensors=True
    #     ).to(device)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        padding_side="left",
    )

    dataset = SupervisedDataset(data_args.data_dir, data_args.dataset, 
                data_args.split, tokenizer, data_args.mode, 
                data_args.randomize_entity_name, 
                model_args.entity_as_new_token,
                model_args.relation_as_new_token,
                5,
                data_args.max_num_rules_per_r,
                data_args.use_demonstrations, 
                data_args.num_positive,
                data_args.num_negative,
                4)

    generation_args = {}
    generation_args["pad_token_id"] = tokenizer.pad_token_id

    if model_args.max_num_reasoning_steps is not None:
        if model_args.max_num_reasoning_steps == 1:
            generation_args["eos_token_id"] = tokenizer('.').input_ids[0]
        else:
            stop_words_ids = [tokenizer(stop_word, return_tensors='pt').input_ids.squeeze() for stop_word in ["."]]
            stopping_criteria = StoppingCriteriaList(
                [StoppingCriteriaSub(stops=stop_words_ids, 
                encounters=model_args.max_num_reasoning_steps)])
            generation_args["stopping_criteria"] = stopping_criteria
    else:
        model_args.max_num_reasoning_steps = 0

    if model_args.decoding_scheme == "sampling":
        generation_args["do_sample"] = True
        generation_args["top_k"] = model_args.generate_n
        generation_args["num_return_sequences"] = model_args.generate_n

    elif model_args.decoding_scheme == "beam_search":
        generation_args["num_beams"] = model_args.generate_n
        generation_args["early_stopping"] = True
        generation_args["num_return_sequences"] = model_args.generate_n

    output = []
    num_correct = 0
    num_wrong = 0
    num_all = 0

    for x_text, y_text, triple in dataset:
        print(x_text, y_text, triple)
        e1, r, e2s = triple
        bad_e2s = dataset.seen_answers[(e1, r)]
        if len(bad_e2s) > 0 and ('countries' not in data_args.dataset) \
            and data_args.dataset != "nell-995" and data_args.split != 'train':
            bad_words = [dataset.grapher.entity2name(e) for e in bad_e2s]
            if model_args.entity_as_new_token:
                bad_words_ids = [tokenizer(w).input_ids for w in bad_words]
            else:
                bad_words_ids = [[tokenizer(' ' + w).input_ids[0]] for w in bad_words]
        else:
            bad_words_ids = []

        encoding = tokenizer(x_text, padding=True, return_tensors='pt').to(device)

        with torch.no_grad():
            if len(bad_words_ids) > 0:
                generated_ids = model.generate(**encoding, **generation_args,
                    bad_words_ids=bad_words_ids, max_length=model_args.max_length)
            else:
                generated_ids = model.generate(**encoding, **generation_args,
                    max_length=model_args.max_length)
        
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        contain_answer = False
        contain_wrong = False
        for text in generated_texts:
            verify = dataset.verify(text, y_text, triple, 
                                model_args.max_num_reasoning_steps)
            print(verify)
            if verify != 'seen':
                if verify == 'correct':
                    contain_answer = True
                elif verify == 'wrong':
                    contain_wrong = True
                output.append((text, y_text))

        if contain_answer:
            num_correct += 1
        if contain_wrong:
            num_wrong += 1
        num_all += 1
        

    if model_args.generate_n > 1:
        print("correct rate: ", num_correct/num_all)
        print("wrong rate: ", num_wrong/num_all)
    else:
        print("Accuracy: ", num_correct/num_all)
    
    
    if data_args.use_demonstrations:
        out_file_name = f'{model_args.output_dir}/{data_args.split}_output_w_demos.txt'
    else:
        out_file_name = f'{model_args.output_dir}/{data_args.split}_output.txt'
    
    with open(out_file_name, 'w') as f:
        for x, y in output:
            if type(y) == list:
                y = '[' + ' '.join(y) + ']'
            f.write(x + '\t' + y + '\n')
        f.write(f"Accuracy: {num_correct/num_all}")
        # if model_args.generate_n > 1:
        #     print("correct rate: ", num_correct/num_all)
        #     print("wrong rate: ", num_wrong/num_all)
        # else:
        #     print("Accuracy: ", num_correct/num_all)

if __name__ == "__main__":
    eval()