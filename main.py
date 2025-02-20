import os
import numpy as np

# =================Config Start================= #

# Load configuration from config.json
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]

edit_num = config["edit_num"]
beta = config["beta"]
alpha = config["alpha"]
nsample = config["nsample"]
model_name = config["model_name"]
dataset_name = config["dataset_name"]
seed = config["seed"]

# =================Config End================= #

from transformers import set_seed
set_seed(seed)
from decker import DecKER
# from model import model, llmtokenizer, contriever, tokenizer
from utils import get_sent_embeddings
import json
from tqdm import tqdm


output_path = f"./output/DecKER_{int(beta*100)}-{int(alpha*10)}__{model_name}_{dataset_name}_{nsample}"

all_dataset = json.load(open(f"./datasets/{dataset_name}.json", "r"))

if edit_num == 0:
    edit_num = len(all_dataset)

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
import json



MODEL_NAME_TO_PATH = {
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2.5-7b":"Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
}

model_path = MODEL_NAME_TO_PATH[model_name]

model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=False,
            torch_dtype=torch.float32 if "qwen2.5-7b" in model_name else torch.float16,
            device_map='auto',
            low_cpu_mem_usage=True)
llmtokenizer = AutoTokenizer.from_pretrained(model_path)

llmtokenizer.pad_token = llmtokenizer.eos_token
llmtokenizer.padding_side = "left"

contriever = AutoModel.from_pretrained('facebook/contriever-msmarco').cuda()
tokenizer = AutoTokenizer.from_pretrained('facebook/contriever-msmarco')

all_dataset_list = [all_dataset[i:i+edit_num] for i in range(0, len(all_dataset), edit_num)]

tot = 0
multiple_cor_base = 0
multiple_cor_bon = 0

for dataset in all_dataset_list:
    new_facts = {}
    target_new_facts = {}
    n2o = {}
    if "pop" in dataset_name or "random" in dataset_name:
        all_facts = set()
        
        for d in dataset:
            r = d["requested_rewrite"][0]
            prompt = d["new_single_hops"][0]["cloze"]
            new_facts[f'{prompt} {r["target_true"]["str"]}'] = f'{prompt} {r["target_new"]["str"]}'
            target_new_facts[f'{prompt} {r["target_new"]["str"]}'] = r["target_new"]["str"]
            n2o[f'{prompt} {r["target_new"]["str"]}'] = f'{prompt} {r["target_true"]["str"]}'
        print(len(new_facts))
        new_facts_list = list(new_facts.values())
    else:
        
        for d in dataset:
            for r in d["requested_rewrite"]:
                new_facts[f'{r["prompt"].format(r["subject"])} {r["target_true"]["str"]}'] = f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}'
                target_new_facts[f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}'] = r["target_new"]["str"]
                n2o[f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}'] = f'{r["prompt"].format(r["subject"])} {r["target_true"]["str"]}'
        print(len(new_facts))

        new_facts_list = list(new_facts.values())

    embs = get_sent_embeddings(new_facts_list, contriever, tokenizer)
    
    results = []
    cate_scores_res = []

    for d in tqdm(dataset):
        
        correct_answers = [d["new_answer"].lower()] + [a.lower() for a in d["new_answer_alias"]]

        flag = False
        bon_flag = False
        tot += 1
        for q in d["questions"]:
            
            result_data = {
                "question": q,
                "correct_answers": correct_answers,
                "right_cot": [a["cloze"] + " " + a["answer"] for a in d["new_single_hops"]],
            }
            result = DecKER(q, model, llmtokenizer, contriever, tokenizer, embs, n2o, new_facts_list, target_new_facts, beta
                            , alpha, num_samples=nsample)
            if result is None:
                results.append(result_data)
                cate_scores_res.append(None)
                continue
            else:
                reses, cate_scores = result
            result_data["res"] = reses
            results.append(result_data)
            cate_scores_res.append(cate_scores)
            

            # DecKER Base Multi-hop Accuracy
            for res in reses:
                if res["mask_cot"]["idx"] >= 2:
                    continue
                if res["new"]["new_answer"]:
                    if res["new"]["new_answer"].lower() in correct_answers:
                        flag = True
                    break
            
            
            # DecKER BoN Multi-hop Accuracy
            is_right = []
            rpp_scores = []
            ki_scores = []
            for res, pe_cate in zip(reses, cate_scores):
                if res["new"]["new_answer"]:
                    is_right.append(res["new"]["new_answer"].lower().strip() in correct_answers)
                    rpp_scores.append(-np.mean(res["mask_cot"]['cate_el']))
                    ki_scores.append(sum([1 if y["right"] else 0 for y in pe_cate]) / len(pe_cate) if pe_cate else 0)
            if len(is_right) == 0:
                continue
            if len(rpp_scores) > nsample // 2:
                max_values, max_indices = torch.topk(torch.tensor(rpp_scores),nsample // 2)
            else:
                max_indices = list(range(len(rpp_scores)))

            idx_both = max_indices[np.argmax([ki_scores[i] for i in max_indices])]
            if is_right[idx_both]:
                bon_flag = True
        
        if flag:
            multiple_cor_base += 1
            
        if bon_flag:
            multiple_cor_bon += 1

        print(f"DecKER Base Multi-hop Accuracy: {multiple_cor_base}/{tot} = {multiple_cor_base/tot}")
        print(f"DecKER BoN Multi-hop Accuracy: {multiple_cor_bon}/{tot} = {multiple_cor_bon/tot}")
        # if tot >= 10:
        #     break

    
if not os.path.exists(f"{output_path}"):
    os.mkdir(f"{output_path}")

        
with open(f"{output_path}/result.json", "w+") as f:
    json.dump(results, f, indent=4)
    
with open(f"{output_path}/cate.json", "w+") as f:
    json.dump(cate_scores_res, f)
