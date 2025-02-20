import os
from tqdm import tqdm
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import random
import numpy as np
from transformers import AutoModel

model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True).to("cuda:0")




def jina_score(texts):
    embeddings = model.encode(texts, task="text-matching")
    return embeddings

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

noedit_data = json.load(open("/PATH/TO/RELATIONS/OF/NO_EDITED/PATH/", "r"))
edit_data = json.load(open("/PATH/TO/RELATIONS/OF/EDITED/PATH/", "r"))

cos_sims = []
for nd, ed in tqdm(zip(noedit_data, edit_data)):
    sim = 0
    for i in range(min(len(nd), len(ed))):
        embeddings = jina_score([nd[i], ed[i]])
        sim += embeddings[0] @ embeddings[1].T
    cos_sims.append(sim / max(len(nd), len(ed)))



print(np.mean(cos_sims))