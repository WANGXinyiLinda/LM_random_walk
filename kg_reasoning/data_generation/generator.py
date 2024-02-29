import json
import os
import numpy as np
import csv
import argparse
import random
from collections import defaultdict
from string import Template
from english_words import get_english_words_set
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
import re
import torch
from tqdm import tqdm
from ast import literal_eval

def load_rule_file(rule_file, min_count = 0):
    rules = {}
    with open(rule_file) as rf:
        raw_rule_dict = json.load(rf)
        for goal in raw_rule_dict:
            rules[int(goal)] = {}
            for rule in raw_rule_dict[goal]:
                if raw_rule_dict[goal][rule]["count"] > min_count:
                    rules[int(goal)][literal_eval(rule)] = raw_rule_dict[goal][rule]
    return rules

class Grapher:
    def __init__(self, data_dir, dataset, randomize=False, use_inverse_r=False,
                 entity_as_new_token=False, relation_as_new_token=False,
                 max_rule_len=2, max_num_rules=100, seed=0):
        self.dataset = dataset
        self.data_dir = data_dir
        self.max_rule_len = max_rule_len
        self.max_num_rules = max_num_rules
        self.use_inverse_r = use_inverse_r
        if self.use_inverse_r:
            self.relation_vocab = json.load(open(f'{data_dir}/{dataset}/vocab/relation_vocab_rev.json'))
            self.triple_file = f'{data_dir}/{dataset}/graph_rev.txt'
        else:
            self.relation_vocab = json.load(open(f'{data_dir}/{dataset}/vocab/relation_vocab.json'))
            self.triple_file = f'{data_dir}/{dataset}/graph.txt'
        self.num_relations = len(self.relation_vocab)
        self.all_relations = list(self.relation_vocab.keys())
        self.entity_vocab = json.load(open(f'{data_dir}/{dataset}/vocab/entity_vocab.json'))
        self.rev_relation_vocab = dict([(v, k) for k, v in self.relation_vocab.items()])
        self.rev_entity_vocab = dict([(v, k) for k, v in self.entity_vocab.items()])

        self.all_rule_probs = None

        self.relation2sent = {}
        self.regex2relation = {}
        r2sent_file = f'{data_dir}/{dataset}/vocab/relation2sent.json'
        if os.path.exists(r2sent_file):
            self.relation2regex = {}
            temp = json.load(open(r2sent_file))
            for r in temp:
                try:
                    r_id = self.relation_vocab[r]
                except:
                    continue
                self.relation2sent[r_id] = temp[r]
                regex_list = []
                for t in temp[r]:
                    regex = t[:-1].replace('$e1', '(?P<e1>.+)').replace('$e2', '(?P<e2>.+)')
                    self.regex2relation[regex] = r_id
                    regex_list.append(regex)
                self.relation2regex[r_id] = regex_list
        else:
            if not relation_as_new_token:
                print("Not use relations as new tokens and no relation templates.")
                exit(1)

        self.rule_weight_file = f'{self.data_dir}/{self.dataset}/rule_probs_mat/{self.max_rule_len}/rules_with_weights.json'
        if os.path.exists(self.rule_weight_file):
            self.rules = load_rule_file(self.rule_weight_file)
        else:
            self.rules = {}
        
        self.gt_rules_file = f'{self.data_dir}/{self.dataset}/gt_rules.json'
        if os.path.exists(self.gt_rules_file):
            self.gt_rules = load_rule_file(self.gt_rules_file)
        else:
            self.gt_rules = {}
            

        self.train_triples = []
        self.possible_goal_e2s = defaultdict(list)
        self.store = defaultdict(list)
        self.path_ranking_predictors = {}

        self.create_graph()
        print("KG constructed")

        self.test_triples = self.load_triples(f'{self.data_dir}/{self.dataset}/test.txt')
        self.dev_triples = self.load_triples(f'{self.data_dir}/{self.dataset}/dev.txt')

        if randomize:
            self.randomize_entity_names()
            print("using randomized entity names...")
        
        self.entity_as_new_token = entity_as_new_token
        self.relation_as_new_token = relation_as_new_token

        random.seed(seed)
        self.seed = seed


    def randomize_entity_names(self):
        save_path = f'{self.data_dir}/{self.dataset}/vocab/randomized_entity_vocab.json'
        if not os.path.isfile(save_path):
            english_words_set = sorted(get_english_words_set(['web2']))
            entity_names = random.choices(english_words_set, k=len(self.entity_vocab))
            # random.shuffle(entity_names)
            # self.random2original_entity = {entity_names[i]: self.rev_entity_vocab[i] 
            #                         for i in range(len(entity_names))}
            self.entity_vocab = {k:v for v, k in enumerate(entity_names)}
            with open(save_path, 'w') as f:
                json.dump(self.entity_vocab, f)
        else:
            self.entity_vocab = json.load(open(save_path))


    def load_triples(self, file_path):
        triples = []
        with open(file_path) as triple_file_raw:
            triple_file = csv.reader(triple_file_raw, delimiter='\t')
            for line in triple_file:
                try:
                    e1 = self.entity_vocab[line[0]]
                    r = self.relation_vocab[line[1]]
                    e2 = self.entity_vocab[line[2]]
                    triples.append((e1, r, e2))
                except:
                    print(f"{line} cannot be parsed")
        return triples
    

    def create_graph(self):
        self.adjacency_matrices = {}

        with open(self.triple_file) as triple_file_raw:
            triple_file = csv.reader(triple_file_raw, delimiter='\t')
            for line in triple_file:
                try:
                    e1 = self.entity_vocab[line[0]]
                    r = self.relation_vocab[line[1]]
                    e2 = self.entity_vocab[line[2]]
                    self.store[e1].append((r, e2))
                except:
                    print(f"{line} cannot be parsed")
                # if 'countries' in self.dataset and r == 0:
                #     self.store[e2].append((2, e1))
                self.train_triples.append((e1, r, e2))
                if e2 not in self.possible_goal_e2s[r]:
                    self.possible_goal_e2s[r].append(e2)

        print("constructing adjacency matrices..")
        sparse_construction = {}
        for e1 in tqdm(self.rev_entity_vocab):
            if len(self.store[e1]) > 0:
                prob = 1/len(self.store[e1])
                for r, e2 in self.store[e1]:
                    if r not in sparse_construction:
                        sparse_construction[r] = {'i': [[e1], [e2]], 'v': [prob]}
                    else:
                        sparse_construction[r]['i'][0].append(e1)
                        sparse_construction[r]['i'][1].append(e2)
                        sparse_construction[r]['v'].append(prob)

        for r in sparse_construction:
            self.adjacency_matrices[r] = torch.sparse_coo_tensor(
                sparse_construction[r]['i'], sparse_construction[r]['v'],
                (len(self.entity_vocab), len(self.entity_vocab)), 
                dtype=torch.float32, device='cuda', check_invariants=True)


    def search_by_rule(self, e1, rule, e2):
        """
        Searches for entities that satisfy a given rule.

        Args:
            e1 (int): The ID of the starting entity.
            rule (tuple of int): The list of relations that define the rule.
            e2 (int): The ID of the target entity.

        Returns:
            list of int: The list of list of entities that satisfy the rule, or an empty list if no entities were found.
        """
        # Initialize the list of entities that satisfy the rule
        result = []
        correct_rate = {True: 0, False: 0}

        # Start the search from the first entity
        queue = [(e1, [])]

        # Use breadth-first search to traverse the graph
        while queue:
            curr, path = queue.pop(0)

            # If we have found the target entity and the path satisfies the rule, add it to the results
            if len(path) == len(rule):
                if curr == e2:
                    result.append([e1] + path)
                    correct_rate[True] += 1
                else:
                    correct_rate[False] += 1

            # If the path length exceeds the length of the rule, terminate the search
            if len(path) >= len(rule):
                continue

            # Otherwise, continue the search by looking for outgoing edges from the current node
            for r, next_node in self.store[curr]:
                # If the next node is not already in the path and the next relation matches the rule, add it to the queue
                if next_node not in path and r == rule[len(path)]:
                    queue.append((next_node, path + [next_node]))

        # Return the list of entities that satisfy the rule, or an empty list if none were found
        return result, correct_rate


    def get_rule_probs(self, rule):
        """
        Searches for probability of all entity pairs that satisfy a given rule.

        Args:
            rule (list of int): a list of relations.

        Returns:
            dict (str: float): probability of arriving at different e2s.
        """
        probs = torch.eye(len(self.entity_vocab), dtype=torch.float32, device='cuda')
        all_probs = []
        for r in rule:
            try:
                probs = torch.sparse.mm(probs, self.adjacency_matrices[r]).to_sparse()
            except:
                print("not able to compute adjacency matrix multiplication on GPU. Try CPU...")
                probs = probs.to('cpu')
                self.adjacency_matrices[r] = self.adjacency_matrices[r].to('cpu')
                probs = torch.sparse.mm(probs, self.adjacency_matrices[r]).to_sparse()
            all_probs.append(probs)
        return all_probs
    
    
    def get_all_rule_probs(self, epsilon=1e-4):
        all_rule_probs = {}
        num_entities = len(self.entity_vocab)
        identity_mat = torch.sparse_coo_tensor([list(range(num_entities))]*2, [1]*num_entities,
                (num_entities, num_entities), dtype=torch.float32, device='cuda', check_invariants=True)
        cur_prods = [([], identity_mat)]

        for i in range(self.max_rule_len):
            next_prods = []
            prob_file_path = f'{self.data_dir}/{self.dataset}/rule_probs_mat/{i+1}/all_rule_probs.pt'
            if os.path.exists(prob_file_path):
                print(f"loading length {i+1} probs...")
                new_probs = torch.load(prob_file_path, map_location=torch.device('cpu'))
                cur_prods = [(k, new_probs[k]) for k in new_probs]
            else:
                os.makedirs(f'{self.data_dir}/{self.dataset}/rule_probs_mat/{i+1}', exist_ok=True)
                print(f"computing length {i+1} probs...")
                new_probs = {}
                for rule, prod in tqdm(cur_prods):
                    for r in self.rev_relation_vocab:
                        if r not in self.adjacency_matrices:
                            continue
                        try:
                            new_prod = torch.sparse.mm(prod, self.adjacency_matrices[r]).to_sparse()
                        except:
                            print("not able to compute adjacency matrix multiplication on GPU. Try CPU...")
                            prod = prod.to('cpu')
                            self.adjacency_matrices[r] = self.adjacency_matrices[r].to('cpu')
                            new_prod = torch.sparse.mm(prod, self.adjacency_matrices[r]).to_sparse()
                        new_rule = list(rule) + [r]
                        if new_prod.sum() > epsilon:
                            next_prods.append((new_rule, new_prod))
                            new_probs[tuple(new_rule)] = new_prod.detach().cpu()
                cur_prods = next_prods
                torch.save(new_probs, prob_file_path)
            all_rule_probs.update(new_probs)

        return all_rule_probs


    def get_path_ranking_data(self, r):
        path_ranking_x = []
        path_ranking_y = []
        num_pos = 0
        num_neg = 0

        if self.all_rule_probs is None:
            self.all_rule_probs = self.get_all_rule_probs()

        all_pos_triples = [(e1, _r, e2) for e1, _r, e2 in self.train_triples if _r==r]
        all_neg_triples = []

        rule_count = {}  
        print("counting rules...")
        for rule in tqdm(self.all_rule_probs):
            c = 0
            for e1, _r, e2 in all_pos_triples:
                c += int(self.all_rule_probs[rule][e1][e2].item() > 0)
            if c > 10:
                rule_count[rule] = c
        
        sorted_counts = sorted(rule_count.items(), key=lambda x: x[1], reverse=True)
        num_rules = min([self.max_num_rules, len(sorted_counts)])
        self.rules[r] = {rule: {'count': c} for rule, c in sorted_counts[:num_rules]}

        print("loading postive examples...")
        for e1, _r, e2 in tqdm(all_pos_triples):
            x = np.zeros(len(self.rules[r]))
            for i, rule in enumerate(self.rules[r]):
                x[i] = self.all_rule_probs[rule][e1][e2]
            if sum(x) > 0:
                path_ranking_x.append(x)
                path_ranking_y.append(1)
                num_pos += 1
                # print(x)
        print("num positive examples: ", num_pos)
        assert len(path_ranking_x) == len(path_ranking_y)

        for e1, _r, e2 in all_pos_triples:
            _, core_neg_e2s, other_neg_e2s = self.sample_negative_e2(e1, r, e2)
            all_neg_triples += [(e1, r, e) for e in core_neg_e2s]
            all_neg_triples += [(e1, r, e) for e in other_neg_e2s]

        print("loading negative examples...")
        # print(all_neg_triples)
        neg_x = []
        for e1, _r, e2 in tqdm(all_neg_triples):
            x = np.zeros(len(self.rules[r]))
            for i, rule in enumerate(self.rules[r]):
                x[i] = self.all_rule_probs[rule][e1][e2]
            # print(x)
            if sum(x) > 0 or self.max_rule_len==1:
                neg_x.append(x)

        if len(neg_x) > num_pos:
            path_ranking_x += random.sample(neg_x, k=num_pos)
            path_ranking_y += [0 for _ in range(num_pos)]
            num_neg = num_pos
        else:
            path_ranking_x += neg_x
            path_ranking_y += [0 for _ in range(len(neg_x))]
            num_neg = len(neg_x)

        print("num negative examples: ", num_neg)
        assert len(path_ranking_x) == len(path_ranking_y)

        path_ranking_x, path_ranking_y = shuffle(
            path_ranking_x, path_ranking_y, random_state=self.seed)
        
        return path_ranking_x, path_ranking_y, num_pos, num_neg

    def get_all_proofs(self, min_count=10):
        self.all_proofs = {}
        for e1, r, e2 in self.train_triples:
            self.all_proofs[(e1,r,e2)] = {}
            if r not in self.rules:
                self.get_path_ranking_data(r)
            for rule in self.rules[r]:
                if self.rules[r][rule]["count"] > min_count:
                    paths, correct_rate = self.search_by_rule(e1, rule, e2)
                    if '#correct' in self.rules[r][rule]:
                        self.rules[r][rule]['#correct'] += correct_rate[True]
                    else:
                        self.rules[r][rule]['#correct'] = correct_rate[True]
                    if '#wrong' in self.rules[r][rule]:
                        self.rules[r][rule]['#wrong'] += correct_rate[False]
                    else:
                        self.rules[r][rule]['#wrong'] = correct_rate[False]
                    if len(paths) > 0:
                        self.all_proofs[(e1,r,e2)][rule] = paths
            if len(self.all_proofs[(e1,r,e2)]) == 0:
                del self.all_proofs[(e1,r,e2)]
        return self.all_proofs
    
        
    def get_rule_weights(self, r):
        '''
        Path ranking algorithm
        '''
        if self.all_rule_probs is None:
            self.all_rule_probs = self.get_all_rule_probs()
        if r not in self.path_ranking_predictors:
            if r in self.rules:
                weights = []
                for rule in self.rules[r]:
                    weights.append(self.rules[r][rule]['weight'])
                weights = np.array(weights)
            else:
                path_ranking_x, path_ranking_y, num_pos, num_neg = self.get_path_ranking_data(r)
                if num_pos > 0 and num_neg > 0:
                    self.path_ranking_predictors[r] = LogisticRegression(random_state=self.seed).fit(
                        path_ranking_x, path_ranking_y)
                    weights = self.path_ranking_predictors[r].coef_[0]
                else:
                    weights = np.ones(len(self.rules[r]))

                rules = list(self.rules[r].keys())
                for rule, w in zip(rules, weights):
                    self.rules[r][rule]['weight'] = w
                print(f"rule weights and counts for {r}: {self.rules[r]}")

                with open(self.rule_weight_file, 'w') as f:
                    _all_rules = {}
                    for goal in self.rules:
                        _all_rules[goal] = {}
                        for rule in self.rules[goal]:
                            _all_rules[goal][str(rule)] = self.rules[goal][rule]
                    json.dump(_all_rules, f)
        else:
            weights = self.path_ranking_predictors[r].coef_[0]

        return weights
    

    def path_ranking_predict(self, triples, r):
        
        weights = self.get_rule_weights(r)
        rules = list(self.rules[r].keys())
        if len(self.gt_rules) > 0:
            gt_rules = list(self.gt_rules[r].keys())
        xs = []
        gt_xs = []
        for e1, _r, e2 in triples:
            assert _r == r
            x = np.zeros(len(rules))
            gt_x = np.zeros(len(rules))
            for i, rule in enumerate(rules):
                x[i] = self.all_rule_probs[rule][e1][e2]
            if self.max_rule_len >= 3 and len(self.gt_rules) > 0:
                for i, rule in enumerate(gt_rules):
                    gt_x[i] = self.all_rule_probs[rule][e1][e2]
            xs.append(list(x))
            gt_xs.append(list(gt_x))
        rule_probs = np.array(xs)
        gt_rule_probs = np.array(gt_xs)
        return np.matmul(rule_probs, weights), np.sum(rule_probs, -1), np.sum(gt_rule_probs, -1)
    

    def entity2name(self, e):
        if self.entity_as_new_token:
            return  f'<entity_{e}>'
        else:
            e_name = self.rev_entity_vocab[e].split('_')
            if 'countries' in self.dataset:
                e_name = [n.capitalize() for n in e_name]
            return ' '.join(e_name)
        

    def name2entity(self, name):
        if self.entity_as_new_token:
            try:
                return int(name.split('_')[-1].split('>')[0])
            except:
                return name
        try:
            return self.entity_vocab['_'.join(name.lower().strip().strip('.').split())]
        except:
            return name
    

    def triple2sent(self, e1, r, e2):
        if self.relation_as_new_token:
            return ' '.join([self.entity2name(e1), f'<relation_{r}>', 
                                self.entity2name(e2)]) + '.'
        if r < self.num_relations:
            temp = Template(random.choice(self.relation2sent[r]))
            return temp.substitute(e1=self.entity2name(e1), 
                                e2=self.entity2name(e2))
        else:
            temp = Template(random.choice(self.relation2sent[r-self.num_relations]))
            return temp.substitute(e1=self.entity2name(e2), 
                                e2=self.entity2name(e1))
        

    def sent2triple(self, sent, r=None):
        if self.relation_as_new_token:
            t = sent.split()
            try:
                e1, r, e2 = t[0].strip(), t[1].strip(), t[2].strip()
                e1 = self.name2entity(e1)
                e2 = self.name2entity(e2)
                r = int(r.split('_')[-1][:-1])
                return (e1, r, e2)
            except:
                return None
        
        triple = None
        if r is None:
            for regex in self.regex2relation:
                m = re.search(regex, sent)
                if m:
                    e1 = self.name2entity(m.group("e1").strip())
                    e2 = self.name2entity(m.group("e2").strip())
                    r = self.regex2relation[regex]
                    triple = (e1, r, e2)
        else:
            for regex in self.relation2regex[r]:
                try:
                    m = re.search(regex, sent)
                except:
                    print(regex)
                    print(sent)
                if m:
                    e1 = self.name2entity(m.group("e1").strip())
                    e2 = self.name2entity(m.group("e2").strip())
                    r = self.regex2relation[regex]
                    triple = (e1, r, e2)
        return triple
    
    
    def proof2sent(self, rule, entities):
        sentences = []
        for i, r in enumerate(rule):
            sent = self.triple2sent(entities[i], r, entities[i+1])
            sentences.append(sent)
        return sentences
    
    
    def sample_negative_e2(self, e1, r, e2):
        core_options = []
        other_options = []
        valid_e2 = [_e for _r, _e in self.store[e1] if _r == r]
        if 'countries' in self.dataset:
            temp = []
            for e in valid_e2:
                temp += [_e for _r, _e in self.store[e] if _r == r]
            valid_e2 += temp
        valid_e2 = set(valid_e2)

        for e in self.rev_entity_vocab:
            if e not in valid_e2 and e is not None:
                if e in self.possible_goal_e2s[r]:
                    core_options.append(e)
                else:
                    other_options.append(e)

        if random.random() < 0.8 and len(core_options) > 0:
            neg_e2 = random.choice(core_options)
        else:
            neg_e2 = random.choice(other_options)

        return neg_e2, core_options, other_options
    

    def get_all_train_data_rule(self, num_positive=3, num_negative=3):
        train_data = []
        for goal in self.all_proofs:
            e1, r, e2 = goal
            for rule in self.all_proofs[goal]:
                for path in self.all_proofs[goal][rule]:
                    for i in range(num_positive):
                        goal_sent = self.triple2sent(e1, r, e2)
                        proof_sents = self.proof2sent(rule, path)
                        reasoning_chain = ' '.join(proof_sents)
                        positive = (goal_sent + ' True or False? ' + reasoning_chain
                                    + ' So ' + goal_sent + ' True.')
                        train_data.append(positive)
                    for i in range(num_negative):
                        neg_e2, _ = self.sample_negative_e2(e1, r, e2)
                        proof_sents = self.proof2sent(rule, path)
                        neg_goal_sent = self.triple2sent(e1, r, neg_e2)
                        goal_sent = self.triple2sent(e1, r, e2)
                        reasoning_chain = ' '.join(proof_sents)
                        negative = (neg_goal_sent + ' True or False? ' + reasoning_chain 
                                    + ' So ' + goal_sent + ' False.')
                        train_data.append(negative)
        return train_data
    

    def get_all_dev_data(self, num_positive=2, num_negative=2):
        dev_triples = self.load_triples(f'{self.data_dir}/{self.dataset}/dev.txt')
        dev_data = []
        for e1, r, e2 in dev_triples:
            for i in range(num_positive):
                goal_sent = self.triple2sent(e1, r, e2)
                if i > 0:
                    while goal_sent == last_goal_sent:
                        goal_sent = self.triple2sent(e1, r, e2)
                last_goal_sent = goal_sent
                positive = goal_sent + ' True or False?'
                dev_data.append((positive, "True."))
            for i in range(num_negative):
                neg_e2, _ = self.sample_negative_e2(e1, r, e2)
                if i > 0:
                    while neg_e2 == last_neg_e2:
                        neg_e2, _ = self.sample_negative_e2(e1, r, e2)
                last_neg_e2 = neg_e2
                negative = self.triple2sent(e1, r, neg_e2) + ' True or False?'
                dev_data.append((negative, "False."))
        return dev_data
    
    
    def get_all_test_data(self, num_positive=2, num_negative=2):
        test_triples = self.load_triples(f'{self.data_dir}/{self.dataset}/test.txt')
        test_data = []
        for e1, r, e2 in test_triples:
            for i in range(num_positive):
                goal_sent = self.triple2sent(e1, r, e2)
                if i > 0:
                    while goal_sent == last_goal_sent:
                        goal_sent = self.triple2sent(e1, r, e2)
                last_goal_sent = goal_sent
                positive = goal_sent + ' True or False?'
                test_data.append((positive, "True."))
            for i in range(num_negative):
                neg_e2, _ = self.sample_negative_e2(e1, r, e2)
                if i > 0:
                    while neg_e2 == last_neg_e2:
                        neg_e2, _ = self.sample_negative_e2(e1, r, e2)
                last_neg_e2 = neg_e2
                negative = self.triple2sent(e1, r, neg_e2) + ' True or False?'
                test_data.append((negative, "False."))
        return test_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="countries_S3")
    args = parser.parse_args()

    grapher = Grapher(data_dir=args.data_dir, dataset=args.dataset)

    out_dir = f'{args.data_dir}/{args.dataset}/generated_text'
    os.makedirs(out_dir, exist_ok=True)

    train_data = grapher.get_all_train_data_random_walk()
    with open(f'{out_dir}/train_sents_reandom_walk.txt', 'w') as f:
        for l in train_data:
            f.write(l + '\n')