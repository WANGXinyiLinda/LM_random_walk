import os
import json
import argparse
import openai

try:
    openai.api_key = os.getenv("OPENAI_API_KEY")
except:
    print("please provide an openai api key as an environment variable")
    exit(1)


def main(args):
    vocab = json.load(open(f'{args.data_dir}/{args.dataset}/vocab/relation_vocab.json'))
    prompt = open(f'{args.data_dir}/{args.dataset}/prompt.txt').read()
    print(prompt)
    out_file = f'{args.data_dir}/{args.dataset}/vocab/relation2sent.json'
    if os.path.exists(out_file):
        relation2sent = json.load(open(out_file))
    else:
        relation2sent = {}
    for r in vocab:
        if r not in relation2sent:
            query = f"Please translate the following knowledge graph triple into three different natural sentences: ($e1, {r}, $e2)."
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query}
                ],
                max_tokens=512
            )
            response = completion.choices[0].message.content
            print(query)
            print(response)
            relation2sent[r] = [res.strip() for res in response.split('\n') if len(res)>0]
            json.dump(relation2sent, open(out_file, 'w'), indent = 4)
    
    for r in relation2sent:
        for sent in relation2sent[r]:
            e1_count = sent.count("$e1")
            e2_count = sent.count("$e2")
            if e1_count != 1 or e2_count != 1:
                print("*** relation ***: ", r)
                print(sent)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="FB15K-237")
    args = parser.parse_args()

    main(args)