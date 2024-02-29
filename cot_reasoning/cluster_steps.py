from tqdm import tqdm
import json, pickle
import collections, time
import argparse
from typing import List, Optional, Tuple, Union
import numpy as np
import transformers
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import sentence_transformers
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import AutoTokenizer, T5EncoderModel

from load_data.preprocess import *
from load_data.supervised_dataset import SupervisedDataset, DataCollatorForSupervisedDataset
from load_data.k_shot_dataset import KshotDataset
from model.vae import VQ_VAE, VAE
from model.utils import model_name_mapping

def extract_step_type(dataset_name:str, model_name_or_path:str, batch_size:int,
                      model_max_length = 1024, train_epoch=10, lr=1e-4,
                      selection_method='k-means',
                      output_dir='extract_steps', cache_dir=None,
                      min_frequency=10, max_frequency=10000, num_types=50,):

    out_dir = f"{output_dir}/{model_name_or_path}/{dataset_name}"
    
    model_name_or_path = model_name_mapping(model_name_or_path)
    
    if model_name_or_path != 'all-mpnet-base-v2':
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, legacy=False)
        tokenizer.model_max_length = model_max_length

    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = 0

    if dataset_name == "gsm8k":
        data_class = GSMData
    elif dataset_name == "math":
        data_class = MathData
    elif dataset_name == "aqua":
        data_class = AquaData
    elif dataset_name == "svamp":
        data_class = SVAMPData
    else:
        raise NotImplementedError
    
    dataset = data_class("train", {})

    if selection_method == 'tf-idf':

        solution_steps = []
        for d in dataset:
            solution_steps += d['y'].split('\n')[:-1]

        solution_steps = [step.strip() for step in solution_steps if len(step.strip())>5 and "The answer is:" not in step]
        solution_steps = np.array(solution_steps)

        save_file = f"{output_dir}/{dataset_name}_{selection_method}.pkl"

        if os.path.isfile(save_file):
            with open(save_file, 'rb') as f:
                vectorizer = pickle.load(f)
            X = vectorizer.transform(solution_steps)
        else:
            vectorizer = TfidfVectorizer(max_df=max_frequency, min_df=min_frequency, tokenizer=word_tokenize)
            X = vectorizer.fit_transform(solution_steps)
            print(X)
            with open(save_file, 'wb') as f:
                pickle.dump(vectorizer, f)

        vocab = vectorizer.get_feature_names_out()
        print(vocab)
        print("num vocab: ", len(vocab))
        ids = np.argmax(X, axis=1)
        # print(ids)
        step_types = []
        for i in ids:
            step_types.append(vocab[i][0][0])
        assert len(step_types) == len(solution_steps)
        all_step_types = set(step_types)
        step_type_count = {}
        for step_type in all_step_types:
            step_type_count[step_type] = step_types.count(step_type)
        print({key: val for key, val in sorted(step_type_count.items(), key = lambda ele: ele[1], reverse = True)})
        print(all_step_types)
        print('num all step types: ', len(all_step_types))

    else:
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        embedding_file = f"{out_dir}/{dataset_name}_embedding.npy"
        text_file = f"{out_dir}/{dataset_name}_text.npy"
        example_id_file = f"{out_dir}/{dataset_name}_example_id.npy"

        if os.path.isfile(embedding_file) and os.path.isfile(text_file):
            step_embeddings = np.load(embedding_file)
            solution_steps = np.load(text_file)
            assert len(step_embeddings) == len(solution_steps)

        else:
            os.makedirs(out_dir, exist_ok=True)
            step_embeddings = []

            embedding_model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name_or_path, trust_remote_code=True, cache_dir=cache_dir,
                torch_dtype=torch.float16, device_map="auto", 
                load_in_8bit=True, offload_folder="offload", 
                offload_state_dict = True)
            embedding_model.eval()
            solution_steps = []
            example_ids = []
            ex_id = 0
            for batch in tqdm(dataloader):
                # print(batch)
                examples = []
                questions = []
                step_text = []
                for j in range(len(batch['x'])):
                    steps = batch['y'][j].strip().split('\n')
                    steps = [step.strip() for step in steps if len(step.strip())>5]
                    if len(steps) > 1:
                        questions.append(batch['x'][j].strip().split('\n'))
                        examples.append(batch['x'][j] + '\n'.join(steps[:-1]))
                        step_text.append(steps)

                inputs = tokenizer(examples, return_tensors="pt", padding="longest",
                    max_length=model_max_length, truncation=True).to('cuda')
                with torch.no_grad():
                    outputs = embedding_model(**inputs, output_hidden_states=True, 
                                return_dict=True)
                # print(outputs)
                last_hidden_states = outputs.hidden_states[-1]
                if 'llama' in model_name_or_path or 'Mistral' in model_name_or_path or 'llemma' in model_name_or_path:
                    split_id = 13
                elif 'gpt2' in model_name_or_path or 'phi-2' in model_name_or_path:
                    split_id = 198
                else:
                    print("step split id not defined for ", model_name_or_path)
                    exit(1)
                step_mask = torch.cumsum(inputs['input_ids']==split_id, dim=-1)
                # print("step mask shape: ", step_mask.size())
                step_mask *= inputs["attention_mask"]
                print(step_mask[0])
                print(len(step_text[0]))
                # print("atten mask shape: ", inputs["attention_mask"].size())
                # print(inputs["attention_mask"][0])
                for hidden, mask, q, steps in zip(last_hidden_states, step_mask, questions, step_text):
                    example_rep = []
                    num_steps = torch.max(mask) + 1
                    start = min(len(q), num_steps-1)
                    # print(start)
                    for j in range(start, num_steps):
                        step_j_mask = (mask == j).int().float()
                        step_j_rep = (hidden * step_j_mask.unsqueeze(-1)).sum(0)
                        step_len = step_j_mask.sum()
                        if step_len > 0:
                            rep = (step_j_rep/step_len).cpu().numpy()
                            if np.isnan(rep).sum() == 0:
                                example_rep.append(rep)
                                solution_steps.append(steps[j-start])
                        else:
                            print("current step is empty")
                    if len(example_rep) > 0:
                        example_rep = np.stack(example_rep, axis=0)
                        step_embeddings.append(example_rep)
                        example_ids += [ex_id for _ in range(len(example_rep))]
                        ex_id += 1
            
            step_embeddings = np.concatenate(step_embeddings, axis=0)
            solution_steps = np.array(solution_steps)
            example_ids = np.array(example_ids)

            assert len(step_embeddings) == len(solution_steps)

            np.save(embedding_file, step_embeddings)
            np.save(text_file, solution_steps)
            np.save(example_id_file, example_ids)

            del embedding_model
        
        with open(f"{out_dir}/{dataset_name}_text.json", 'w') as wf:
            json.dump(solution_steps.tolist(), wf)
        
        out_dir = f"{out_dir}/{selection_method}-k={num_types}"

        if selection_method == 'k-means':
            print(step_embeddings)
            cluster_model_file = f"{out_dir}/{dataset_name}_{selection_method}_{num_types}.pkl"
            if os.path.isfile(cluster_model_file):
                with open(cluster_model_file, 'rb') as f:
                    cluster_model = pickle.load(f)
            else:
                os.makedirs(f"{out_dir}", exist_ok=True)
                step_embeddings = np.float32(step_embeddings)
                cluster_model = KMeans(n_clusters=num_types, n_init=10, random_state=0).fit(step_embeddings)
                print(cluster_model)
                print(cluster_model.labels_)
                print(cluster_model.cluster_centers_)
            
                with open(cluster_model_file, 'wb') as f:
                    pickle.dump(cluster_model, f)

            all_preds = cluster_model.labels_
                    
            assert len(all_preds) == len(solution_steps)

            np.save(f"{out_dir}/clusters.npy", all_preds)

            step_ids = np.arange(len(solution_steps))

            for i in range(num_types):
                print(f"cluster {i}: ", np.sum(cluster_model.labels_==i))
                with open(f"{out_dir}/{dataset_name}_{num_types}_{i}.txt", 'w') as f:
                    f.write('\n'.join(list(solution_steps[cluster_model.labels_==i])))

            # print(cluster_model.get_feature_names_out())

        if selection_method == 'balanced-k-means':
            # To install, see https://github.com/kernelmachine/balanced-kmeans/tree/main
            from kmeans_pytorch import KMeans as BalancedKMeans
            cluster_model_file = f"{out_dir}/{dataset_name}_{selection_method}_{num_types}.pkl"
            if os.path.isfile(cluster_model_file):
                with open(cluster_model_file, 'rb') as f:
                    cluster_model = pickle.load(f)
            else:
                os.makedirs(f"{out_dir}", exist_ok=True)
                step_embeddings = torch.from_numpy(np.float32(np.concatenate(step_embeddings, axis=0))).cuda()
                cluster_model = BalancedKMeans(
                    n_clusters=num_types, 
                    device='cuda', 
                    balanced=True, 
                )
                balanced_labels = cluster_model.fit(step_embeddings, iter_limit=300, tol=0.).numpy()
                cluster_model_ub = BalancedKMeans(
                    n_clusters=num_types, 
                    device='cuda', 
                    balanced=False, 
                )
                unbalanced_labels = cluster_model_ub.fit(step_embeddings, iter_limit=50, tol=0.).numpy()

                def get_distances(centroids, X):
                    assert centroids.size(1) == X.size(1)
                    assert centroids.ndim == X.ndim == 2
                    return (centroids.unsqueeze(0) - X.unsqueeze(1)).pow(2).sum(-1)
 
                balanced_dist = get_distances(cluster_model.cluster_centers, step_embeddings)
                unbalanced_dist = get_distances(cluster_model_ub.cluster_centers, step_embeddings)

                print(f'average dist to closest cluster for BALANCED   : {balanced_dist.min(1)[0].mean().item():.4f}')
                print(f'average dist to closest cluster for UNBALANCED : {unbalanced_dist.min(1)[0].mean().item():.4f}')

                # wrap it inside the regular k-means module to piggyback on existing code
                dummy_input = step_embeddings[:10].cpu().numpy()
                sklearn_cluster_model = KMeans(n_clusters=num_types, n_init=10, random_state=0).fit(dummy_input)
                sklearn_cluster_model.cluster_centers_ = cluster_model.cluster_centers.cpu().numpy()

                assert np.all(
                    sklearn_cluster_model.predict(step_embeddings.cpu().numpy()) ==
                    cluster_model.predict(step_embeddings).cpu().numpy()
                )
                with open(cluster_model_file, 'wb') as f:
                    pickle.dump(sklearn_cluster_model, f)
        
            if len(balanced_labels) != len(solution_steps):
                print("num text: ", len(solution_steps))
                print("num embeddings: ", len(balanced_labels))

            for i in range(num_types):
                print(f"cluster {i}: ", np.sum(balanced_labels==i))
                if len(balanced_labels) == len(solution_steps):
                    with open(f"{out_dir}/{dataset_name}_{num_types}_{i}.txt", 'w') as f:
                        f.write('\n'.join(list(solution_steps[balanced_labels==i])))

        elif 'vae' in selection_method:

            out_dir = f"{out_dir}-lr={lr}-batch={batch_size}-epoch={train_epoch}"
            cluster_model_dir = f"{out_dir}/epoch{train_epoch-1}"
            cluster_model_file = f"{cluster_model_dir}/{dataset_name}_{selection_method}_{num_types}.pkl"
            # step_embeddings = torch.from_numpy(step_embeddings).float().to('cuda')

            if os.path.isfile(cluster_model_file):
                print("model exists, loading...")
                with open(cluster_model_file, 'rb') as f:
                    cluster_model = pickle.load(f)

                print("inspecting clusters...")
                indices = []
                cluster_model.eval()
                with torch.no_grad():
                    for i in tqdm(range(len(step_embeddings)//batch_size)):
                        input_batch = step_embeddings[i*batch_size: 
                            min((i+1)*batch_size, len(step_embeddings))]
                        input_batch = torch.from_numpy(input_batch).float().to('cuda')
                        idx = cluster_model.predict(input_batch)
                        indices.append(idx.cpu().numpy())
                indices = np.concatenate(indices, axis=0)
                assert len(indices) == len(solution_steps)

                step_ids = np.arange(len(solution_steps))

                for i in range(num_types):
                    print(f"cluster {i}: ")
                    print(np.sum(indices == i))
                    with open(f"{cluster_model_dir}/{dataset_name}_{num_types}_{i}.txt", 'w') as f:
                        f.write('\n'.join(list(solution_steps[indices == i])))
                    np.save(f"{out_dir}/cluster={i}.npy", step_ids[indices==i])
            else:
                os.makedirs(f"{out_dir}", exist_ok=True)
                input_dim = step_embeddings.shape[1]
                print("input embedding dim: ", input_dim) 
                neg_cost = 0
                if 'next-step' in selection_method:
                    loss_type = 'next-step'
                elif 'contrastive' in selection_method:
                    loss_type = 'contrastive'
                    neg_cost = 0.1
                else:
                    loss_type = 'reconstruct'

                cluster_model = VAE(input_size=input_dim, num_embeddings=num_types,
                                    neg_cost=neg_cost, loss_type=loss_type, lr=lr, 
                                    embedding_dim=64, 
                                    hidden_size=512, 
                                    num_layers=3,
                                    dropout=0.0).to('cuda')
                checkpoint_vals = collections.defaultdict(lambda: [])
                all_results = []
                for epoch in range(train_epoch):
                    epoch_start_time = time.time()
                    cluster_model_dir = f"{out_dir}/epoch{epoch}"
                    cluster_model_file = f"{out_dir}/epoch{epoch}/{dataset_name}_{selection_method}_{num_types}.pkl"
                    os.makedirs(cluster_model_dir, exist_ok=True)
                    print(f"training epoch {epoch+1}...")
                    np.random.shuffle(step_embeddings)
                    for i in tqdm(range(len(step_embeddings)//batch_size)):
                        input_batch = step_embeddings[i*batch_size: 
                            min((i+1)*batch_size, len(step_embeddings))]
                        # print("input batch: ", input_batch)
                        input_batch = torch.from_numpy(input_batch).float().to('cuda')
                        losses = cluster_model.update(input_batch)
                        for key, val in losses.items():
                            checkpoint_vals[key].append(val)
                    # print(losses)
                    print(f"epoch {epoch+1} finished, time: {time.time()-epoch_start_time}")
                    checkpoint_vals['epoch_time'].append(time.time() - epoch_start_time)

                    print("saving epoch checkpoint...")
                    results = {'epoch': epoch+1}
                    for key, val in checkpoint_vals.items():
                        results[key] = np.mean(val)
                    print(results)
                    results['args'] = vars(args)
                    all_results.append(results)

                    epochs_path = os.path.join(out_dir, 'results.jsonl')
                    with open(epochs_path, 'w') as f:
                        f.write(json.dumps(all_results, sort_keys=True) + "\n")

                    checkpoint_vals = collections.defaultdict(lambda: [])
                    with open(cluster_model_file, 'wb') as f:
                        pickle.dump(cluster_model, f)

                    print("inspecting clusters...")
                    indices = []
                    cluster_model.eval()
                    with torch.no_grad():
                        for i in tqdm(range(len(step_embeddings)//batch_size)):
                            input_batch = step_embeddings[i*batch_size: 
                                min((i+1)*batch_size, len(step_embeddings))]
                            input_batch = torch.from_numpy(input_batch).float().to('cuda')
                            idx = cluster_model.predict(input_batch)
                            indices.append(idx.cpu().numpy())
                    all_preds = np.concatenate(indices, axis=0)

                    assert len(all_preds) == len(solution_steps)

                    step_ids = np.arange(len(solution_steps))
                    np.save(f"{cluster_model_dir}/clusters.npy", all_preds)                

                    for i in range(num_types):
                        print(f"cluster {i}: ")
                        print(np.sum(all_preds == i))
                        with open(f"{cluster_model_dir}/{dataset_name}_{num_types}_{i}.txt", 'w') as f:
                            f.write('\n'.join(list(solution_steps[all_preds == i])))

        tsne_file = f"{out_dir}/tsne.npy"

        if os.path.isfile(tsne_file):
            X = np.load(tsne_file)
        else:
            X = TSNE(n_components=2, learning_rate='auto',
                            init='random', perplexity=3).fit_transform(np.float32(step_embeddings))
            np.save(tsne_file, X)

        plt.scatter(X[:, 0], X[:, 1], c=all_preds, s=2, cmap='viridis')
        plt.title(f"Number of Clusters = {num_types}")
        plt.savefig(f"{out_dir}/kmeans.png")
        plt.show()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gsm8k', help='dataset name')
    parser.add_argument('--model_name_or_path', type=str, default='meta-llama/Llama-2-7b-hf', help='model name or path')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--model_max_length', type=int, default=2048, help='model max length')
    parser.add_argument('--selection_method', type=str, default='k-means', 
                        choices=['tf-idf', 'k-means', 'vae-next-step', 'vae-contrastive',
                                 'vae', 'balanced-k-means'])
    parser.add_argument('--output_dir', type=str, default='load_data/extract_steps', help='output dir')
    parser.add_argument('--cache_dir', type=str, default=None, help='cache dir')
    parser.add_argument('--min_frequency', type=int, default=0.05, help='min frequency')
    parser.add_argument('--max_frequency', type=int, default=0.80, help='max frequency')
    parser.add_argument('--num_types', type=int, default=50, help='number of reasoning types')
    parser.add_argument('--train_epoch', type=int, default=100, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

    args = parser.parse_args()

    extract_step_type(args.dataset, args.model_name_or_path, args.batch_size,
                      args.model_max_length, args.train_epoch, args.lr,
                      args.selection_method, args.output_dir, 
                      args.cache_dir, args.min_frequency, args.max_frequency,
                      args.num_types)