from dataclasses import dataclass, field
from typing import Optional
import transformers
import torch
import numpy as np
import scipy
import os, json
import ast
import seaborn as sn 
import matplotlib.pyplot as plt

from eval import SupervisedDataset

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="gpt2")
    cache_dir: Optional[str] = field(default=None)
    output_dir: Optional[str] = field(default=None, metadata={"help": "Path to the output dir."})
    max_length: Optional[int] = field(default=128)
    decoding_scheme: Optional[str] = field(default="greedy")
    max_num_reasoning_steps: Optional[int] = field(default=None)
    generate_n: Optional[int] = field(default=10)
    generate: Optional[bool] = field(default=False)
    entity_as_new_token: Optional[bool] = field(default=True)
    relation_as_new_token: Optional[bool] = field(default=True)
    max_rule_len: Optional[int] = field(default=5)
    max_num_rules: Optional[int] = field(default=100)
    num_neg: Optional[int] = field(default=5)
    do_pra: Optional[bool] = field(default=True)
    do_hidden_pred: Optional[bool] = field(default=False)
    do_logistic_regression: Optional[bool] = field(default=False)
    restricted_vocab: Optional[str] = field(default='all_entities',
                        metadata={"choices": ["core_entities", "all_entities"]})
    seed: Optional[int] = field(default=1234)
    pra_temp: Optional[float] = field(default=100)
    load_in_16bits: Optional[bool] = field(default=False)

@dataclass
class DataArguments:
    data_dir: str = field(default='data', metadata={"help": "Path to the training data."})
    dataset: str = field(default='countries_S3', metadata={"help": "dataset name."})
    batch_size: Optional[int] = field(default=32)
    split: Optional[str] = field(default="test")
    use_demonstrations: Optional[bool] = field(default=False)
    mode: str = field(default="completion")
    randomize_entity_name: Optional[bool] = field(default=False)
    num_examples: Optional[int] = field(default=-1)

def KL(P, Q, epsilon = 1e-4):
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0. """
    assert len(P) == len(Q)
    if np.isnan(P).any() or np.isnan(Q).any():
        return 1/epsilon
    kl = 0.0
    for p, q in zip(P, Q):
        if p > epsilon:
            kl += p * np.log(p / (q + epsilon))
    return kl

def aggragate(ps, ids):
    aggragated_ps = {}
    for p, i in zip(ps, ids):
        if i in aggragated_ps:
            aggragated_ps[i] += p
        else:
            aggragated_ps[i] = p
    unique_ids = list(aggragated_ps.keys())
    _ps = [aggragated_ps[i] for i in unique_ids]
    return np.array(unique_ids), np.array(_ps)

def parse_outputs(file):
    results = []
    with open(file) as rf:
        for l in rf:
            if 'hidden layer predictions:' in l:
                r = l.strip().split('hidden layer predictions:')[-1].strip()
                r = ast.literal_eval(r)
                results.append(r)
    return results
    

if __name__ == "__main__":

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_args.output_dir is None:
        model_args.output_dir = model_args.model_name_or_path
    else:
        os.makedirs(model_args.output_dir, exist_ok = True)

    kld_heat_maps = {'uniform': [], 'groundtruth': [], 'PRA': [], 'sum': [], 'gt_rule': []}
    hypotheses = ['uniform', 'groundtruth', 'PRA', 'sum', 'gt_rule']
    heat_map_json = f'{model_args.output_dir}/klds.json'
    path_dist_json = f'{model_args.output_dir}/pra_dist.json'

    if os.path.exists(heat_map_json):
        kld_heat_maps = json.load(open(heat_map_json))
    else:
        all_path_dist = {}
        if os.path.exists(path_dist_json):
            temp = json.load(open(path_dist_json))
            for i in temp:
                all_path_dist[int(i)] = temp[i]

        for lm_path_len in range(1, 11):
            model_name_or_path = f"{model_args.model_name_or_path}/{lm_path_len}"
            if model_args.load_in_16bits:
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    cache_dir=model_args.cache_dir,
                    device_map="auto", torch_dtype=torch.float16
                ).to(device)
            else:
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    cache_dir=model_args.cache_dir,
                ).to(device)

            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name_or_path,
                cache_dir=model_args.cache_dir,
                padding_side="left",
                use_fast=False
            )

            for h in hypotheses:
                kld_heat_maps[h].append([])

            all_llm_dist = None

            for pra_path_len in range(1, model_args.max_rule_len + 1):

                if pra_path_len in all_path_dist:
                    cur_dists = all_path_dist[pra_path_len]
                else:
                    cur_dists = []

                dataset = SupervisedDataset(data_args.data_dir, data_args.dataset, 
                        data_args.split, tokenizer, data_args.mode,
                        data_args.randomize_entity_name,
                        model_args.entity_as_new_token, model_args.relation_as_new_token,
                        pra_path_len, model_args.max_num_rules, 
                        data_args.use_demonstrations, k=4, seed=model_args.seed) 

                out_file_name = f'{model_args.output_dir}/{lm_path_len}/{data_args.split}'
                os.makedirs(f'{model_args.output_dir}/{lm_path_len}', exist_ok = True)

                if model_args.restricted_vocab == "core_entities":
                    out_file_name += '_core'
                elif model_args.restricted_vocab == "all_entities":
                    out_file_name += '_all'
                else:
                    print("no such vocab.")
                    exit(1)
                if model_args.do_pra:
                    out_file_name += f'_path_ranking_len={pra_path_len}_temp={model_args.pra_temp}'
                if model_args.do_hidden_pred:
                    out_file_name += f'_hidden_pred'

                out_file_name += '.txt'
                wf = open(out_file_name, 'w')

                pra_num_correct = 0
                vanilla_num_correct = 0
                num_all = 0
                llm_num_correct = 0
                klds = {}

                if all_llm_dist is None:

                    all_llm_dist = []

                    for idx, (x_text, y_text, triple) in enumerate(dataset):
                        e1, r, e2s = triple
                        seen_e2s = dataset.seen_answers[(e1, r)]
                        num_all += 1
                        dist = {}

                        if 'countries_S3' == data_args.dataset:
                            e2 = e2s[0]
                            e2s = []
                            for e in dataset.grapher.possible_goal_e2s[r]:
                                for _r, _e in dataset.grapher.store[e]:
                                    if _r == r and _e == e2:
                                        e2s.append(_e)

                        wf.write(f"prompt: {e1} {r}\n")
                        wf.write(f"possible answers: {' '.join([str(e) for e in e2s])}\n")
                        wf.write(f"seen answers: {' '.join([str(e) for e in seen_e2s])}\n")

                        triples = []
                        token_ids = []
                        if model_args.restricted_vocab == "core_entities":
                            all_e2s = np.array(dataset.grapher.possible_goal_e2s[r])
                        elif model_args.restricted_vocab == "all_entities":
                            all_e2s = np.array(list(dataset.grapher.rev_entity_vocab.keys()))
                        else:
                            print("no such vocab.")
                            exit(1)        

                        for _e2 in all_e2s:
                            triples.append((e1, r, _e2))
                            name = dataset.grapher.entity2name(_e2)
                            if not model_args.entity_as_new_token:
                                name = ' ' + name
                            token_ids.append(tokenizer(name).input_ids[0])
                        
                        token_ids = np.array(token_ids)
                        
                        if pra_path_len in all_path_dist:
                            dist = cur_dists[idx]

                        else:
                            dist['uniform'] = np.ones(len(all_e2s))/len(all_e2s)
                            dist['groundtruth'] = np.zeros(len(all_e2s))
                            for i, e in enumerate(all_e2s):
                                if e in e2s:
                                    dist['groundtruth'][i] = 1/len(e2s)

                            pra_scores, vanilla_scores, gt_scores = \
                                dataset.grapher.path_ranking_predict(triples, r)
                            dist['PRA'] = scipy.special.softmax(pra_scores*model_args.pra_temp)
                            dist['sum'] = scipy.special.softmax(vanilla_scores*model_args.pra_temp)
                            dist['gt_rule'] = scipy.special.softmax(gt_scores*model_args.pra_temp)

                            if not model_args.entity_as_new_token:
                                new_token_ids, dist['PRA'] = aggragate(dist['PRA'], token_ids)
                                new_token_ids, vanilla_dist = aggragate(vanilla_dist, token_ids)
                                token_ids = new_token_ids
                            
                            cur_dists.append({k: list(dist[k]) for k in dist})

                            sorted_pos = np.argsort(-pra_scores)
                            sorted_toks = token_ids[sorted_pos]

                            for j, tok in enumerate(sorted_toks):
                                tok_str = tokenizer.decode(tok)
                                pra_pred = dataset.grapher.name2entity(tok_str)
                                if pra_pred not in seen_e2s:
                                    break

                            if 'countries' in data_args.dataset:
                                for _r, _e in dataset.grapher.store[pra_pred]:
                                    if _r == r and _e in e2s:
                                        pra_num_correct += 1
                                        break
                            elif pra_pred in e2s:
                                pra_num_correct += 1

                            sorted_pos = np.argsort(-vanilla_scores)
                            sorted_toks = token_ids[sorted_pos]

                            for j, tok in enumerate(sorted_toks):
                                tok_str = tokenizer.decode(tok)
                                vanilla_pred = dataset.grapher.name2entity(tok_str)
                                if vanilla_pred not in seen_e2s:
                                    break

                            if 'countries' in data_args.dataset:
                                for _r, _e in dataset.grapher.store[vanilla_pred]:
                                    if _r == r and _e in e2s:
                                        vanilla_num_correct += 1
                                        break
                            elif vanilla_pred in e2s:
                                vanilla_num_correct += 1

                            wf.write('ground truth: ' +  x_text + ' [' + ', '.join(y_text) + ']\n')

                        encoding = tokenizer(x_text+ ' ', 
                                            padding=True, return_tensors='pt').to(device)
                        with torch.no_grad():
                            outputs = model(**encoding, labels=encoding["input_ids"])
                        
                        score = outputs.logits[0][-1].detach().cpu().numpy()
                        entity_score = score[token_ids]
                        llm_dist = scipy.special.softmax(entity_score)
                        all_llm_dist.append(llm_dist)

                        sorted_pos = np.argsort(-entity_score)
                        sorted_e2s = all_e2s[sorted_pos]
                        cur_kld = {}

                        for k in hypotheses:
                            cur_kld[k] = KL(dist[k], llm_dist)
                            if k in klds:
                                klds[k].append(cur_kld[k])
                            else:
                                klds[k] = [cur_kld[k]]
                        
                        for j, llm_pred in enumerate(sorted_e2s):
                            if llm_pred not in seen_e2s:
                                break
                    
                        correct = False
                        if llm_pred in e2s:
                            llm_num_correct += 1
                            wf.write(f"correct answer\n")
                            correct = True
                        if not correct:
                            wf.write(f"wrong answer\n")
                        
                        wf.write(f"LM prediction: {llm_pred}\n")

                    wf.write('\n')
                
                else:
                    if pra_path_len in all_path_dist:
                        for llm_dist, dist in zip(all_llm_dist, all_path_dist[pra_path_len]):
                            cur_kld = {}
                            for k in hypotheses:
                                cur_kld[k] = KL(dist[k], llm_dist)
                                if k in klds:
                                    klds[k].append(cur_kld[k])
                                else:
                                    klds[k] = [cur_kld[k]]
                    else:
                        for idx, (x_text, y_text, triple) in enumerate(dataset):
                            e1, r, e2s = triple
                            seen_e2s = dataset.seen_answers[(e1, r)]
                            num_all += 1
                            dist = {}

                            if 'countries_S3' == data_args.dataset:
                                e2 = e2s[0]
                                e2s = []
                                for e in dataset.grapher.possible_goal_e2s[r]:
                                    for _r, _e in dataset.grapher.store[e]:
                                        if _r == r and _e == e2:
                                            e2s.append(_e)

                            wf.write(f"prompt: {e1} {r}\n")
                            wf.write(f"possible answers: {' '.join([str(e) for e in e2s])}\n")
                            wf.write(f"seen answers: {' '.join([str(e) for e in seen_e2s])}\n")

                            triples = []
                            token_ids = []
                            if model_args.restricted_vocab == "core_entities":
                                all_e2s = np.array(dataset.grapher.possible_goal_e2s[r])
                            elif model_args.restricted_vocab == "all_entities":
                                all_e2s = np.array(list(dataset.grapher.rev_entity_vocab.keys()))
                            else:
                                print("no such vocab.")
                                exit(1)         
                            
                            dist['uniform'] = np.ones(len(all_e2s))/len(all_e2s)
                            dist['groundtruth'] = np.zeros(len(all_e2s))
                            for i, e in enumerate(all_e2s):
                                if e in e2s:
                                    dist['groundtruth'][i] = 1/len(e2s)

                            wf.write(f"groundtruth distribution: {dist['groundtruth']} \n")

                            for _e2 in all_e2s:
                                triples.append((e1, r, _e2))
                                name = dataset.grapher.entity2name(_e2)
                                if not model_args.entity_as_new_token:
                                    name = ' ' + name
                                token_ids.append(tokenizer(name).input_ids[0])
                            
                            token_ids = np.array(token_ids)

                            pra_scores, vanilla_scores, gt_scores = \
                                dataset.grapher.path_ranking_predict(triples, r)
                            dist['PRA'] = scipy.special.softmax(pra_scores*model_args.pra_temp)
                            dist['sum'] = scipy.special.softmax(vanilla_scores*model_args.pra_temp)
                            dist['gt_rule'] = scipy.special.softmax(gt_scores*model_args.pra_temp)

                            if not model_args.entity_as_new_token:
                                new_token_ids, dist['PRA'] = aggragate(dist['PRA'], token_ids)
                                new_token_ids, vanilla_dist = aggragate(vanilla_dist, token_ids)
                                token_ids = new_token_ids

                            cur_dists.append({k: list(dist[k]) for k in dist})

                            sorted_pos = np.argsort(-pra_scores)
                            sorted_toks = token_ids[sorted_pos]

                            for j, tok in enumerate(sorted_toks):
                                tok_str = tokenizer.decode(tok)
                                pra_pred = dataset.grapher.name2entity(tok_str)
                                if pra_pred not in seen_e2s:
                                    break

                            if 'countries' in data_args.dataset:
                                for _r, _e in dataset.grapher.store[pra_pred]:
                                    if _r == r and _e in e2s:
                                        pra_num_correct += 1
                                        break
                            elif pra_pred in e2s:
                                pra_num_correct += 1

                            sorted_pos = np.argsort(-vanilla_scores)
                            sorted_toks = token_ids[sorted_pos]

                            for j, tok in enumerate(sorted_toks):
                                tok_str = tokenizer.decode(tok)
                                vanilla_pred = dataset.grapher.name2entity(tok_str)
                                if vanilla_pred not in seen_e2s:
                                    break

                            if 'countries' in data_args.dataset:
                                for _r, _e in dataset.grapher.store[vanilla_pred]:
                                    if _r == r and _e in e2s:
                                        vanilla_num_correct += 1
                                        break
                            elif vanilla_pred in e2s:
                                vanilla_num_correct += 1

                            wf.write('ground truth: ' +  x_text + ' [' + ', '.join(y_text) + ']\n')

                            cur_kld = {}
                            for k in hypotheses:
                                cur_kld[k] = KL(dist[k], all_llm_dist[idx])
                                if k in klds:
                                    klds[k].append(cur_kld[k])
                                else:
                                    klds[k] = [cur_kld[k]]

                if pra_path_len not in all_path_dist:
                    all_path_dist[pra_path_len] = cur_dists

                    pra_acc = pra_num_correct/num_all
                    print(f"PRA acc: {pra_acc}")
                    wf.write(f"PRA acc: {pra_acc}\n")

                    vanilla_acc = vanilla_num_correct/num_all
                    print(f"Vanilla acc: {vanilla_acc}")
                    wf.write(f"Vanilla acc: {vanilla_acc}\n")
                
                if all_llm_dist is None:
                    llm_acc = llm_num_correct/num_all
                    print(f"LLM acc: {llm_acc}")
                    wf.write(f"LLM acc: {llm_acc}\n")

                for h in hypotheses:
                    avg_kld = np.mean(klds[h])
                    std_kld = np.std(klds[h])
                    print(f"    {h} KLD average: {avg_kld} +- {std_kld}")
                    wf.write(f"    {h} KLD average: {avg_kld} +- {std_kld}\n")
                    kld_heat_maps[h][-1].append(avg_kld)

                with open(path_dist_json, 'w') as wf:
                    json.dump(all_path_dist, wf)

        with open(heat_map_json, 'w') as wf:
            json.dump(kld_heat_maps, wf)

    for h in hypotheses:
        data = np.array(kld_heat_maps[h])
        # sn.set(font_scale=1.05)
        hm = sn.heatmap(data=data, xticklabels=range(1, data.shape[1]+1), yticklabels=range(1, data.shape[0]+1), annot=True)
        l = h[0].upper() + h[1:]
        plt.xlabel(f'{l} Distribution', fontsize=11)
        plt.ylabel('LM Distribution', fontsize=11)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.savefig(f"{model_args.output_dir}/{h}.png")
        plt.close()