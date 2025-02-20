
from tqdm import tqdm
import torch
import json
import torch.nn.functional as F



def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def get_sent_embeddings(sents, contriever, tok, BSZ=32):    
    all_embs = []
    for i in tqdm(range(0, len(sents), BSZ)):
        sent_batch = sents[i:i+BSZ]
        inputs = tok(sent_batch, padding=True, truncation=True, return_tensors='pt').to("cuda")
        with torch.no_grad():
            outputs = contriever(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
        all_embs.append(embeddings.cpu())
    all_embs = torch.vstack(all_embs)
    return all_embs

def retrieve_facts(query, fact_embs, contriever, tok, k=2):
    inputs = tok([query], padding=True, truncation=True, return_tensors='pt').to("cuda")
    with torch.no_grad():
        outputs = contriever(**inputs)
        query_emb = mean_pooling(outputs[0], inputs['attention_mask']).cpu()
    sim = (query_emb @ fact_embs.T)[0]
    knn = sim.topk(k, largest=True)
    return knn.indices.tolist(), knn.values.tolist()

#------------------------------------------------------------#
mask_prompts = json.load(open("./prompt/mask.json"))
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
class MaxScoreLogitsProcessor(LogitsProcessor):
    def __init__(self):
        super().__init__()

    def __call__(self, input_ids, scores) -> torch.Tensor:
        max_idx = torch.argmax(scores[0])
        # scores[0][0:max_idx] = -float("inf")
        # scores[0][max_idx+1:] = -float("inf")
        new_scores = scores.clone()
        new_scores[0][0:max_idx] = -float("inf")
        new_scores[0][max_idx+1:] = -float("inf")
        return new_scores
logits_processor = LogitsProcessorList([MaxScoreLogitsProcessor()])

def get_mask(q, model, llmtokenizer, num_return_sequences=1):
    msg = mask_prompts + [{"role": "user", "content": q}]

    generation_config = dict(
                            do_sample=True,
                            max_new_tokens=100,
                            return_dict_in_generate=True,
                            output_hidden_states=True,
                            output_logits = True,
                            output_scores = True,
                            top_p=0.95,
                            temperature=1.2,
                            num_return_sequences=num_return_sequences+1)
        
    input_ids = llmtokenizer.apply_chat_template(
        msg,
        add_generation_prompt=True,
    )
    
    input_ids += llmtokenizer.encode("[STEP] ", add_special_tokens=False)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
    
    outputs = model.generate(
        input_ids,
        pad_token_id=llmtokenizer.eos_token_id,
        attention_mask=input_ids.ne(llmtokenizer.eos_token_id),
        logits_processor=logits_processor,
        **generation_config
    )
    with torch.no_grad():
        model_name = model.__class__.__name__
        if "llama" in model_name.lower():
            end_token_ids = [128001, 128009]
        elif "qwen" in model_name.lower():
            end_token_ids = [151645,151643]
        elif "olmo" in model_name.lower():
            end_token_ids = [100257, 100265]
        start_cate = llmtokenizer.encode("[CATEGORY", add_special_tokens=False)
        # pl, el = [[]]*num_return_sequences, [[]]*num_return_sequences
        # pl_nocate, el_nocate = [[]]*num_return_sequences, [[]]*num_return_sequences
        pl, el, pl_nocate, el_nocate = [], [], [], []
        completed = []
        for _ in range(num_return_sequences):
            pl.append([])
            el.append([])
            pl_nocate.append([])
            el_nocate.append([])
            completed.append(False)
        
        
        outputs_ = []
        for i in range(num_return_sequences):
            response = outputs.sequences[i][input_ids.shape[-1]:]   
            output_ = llmtokenizer.decode(response).strip()
            outputs_.append(output_)
        
        # for i in range(len(outputs.logits)):
        #     now_id = outputs.sequences[0][input_ids.shape[-1]+i].item()
            
        for i in range(len(outputs.logits)):
            probabilities = F.softmax(outputs.logits[i], dim=-1)
            # log_probabilities = torch.log(probabilities)
            # entropy = -probabilities * log_probabilities
            # entropy_sum2 = torch.sum(entropy, dim=-1)
            entropy = torch.special.entr(probabilities)
            entropy_sum = torch.sum(entropy, dim=-1)
            
            # print(entropy_sum)
            
            for j in range(num_return_sequences):
                now_id = outputs.sequences[j][input_ids.shape[-1]+i].item()
                # print(llmtokenizer.decode([now_id]))
                if i < len(outputs.logits) - len(start_cate) and outputs.sequences[j][input_ids.shape[-1]+i:input_ids.shape[-1]+i+len(start_cate)].tolist() == start_cate:
                    completed[j] = True

                if now_id in end_token_ids:
                    continue
                if not completed[j]:
                    pl_nocate[j].append(probabilities[j][now_id].item())
                    el_nocate[j].append(entropy_sum[j].item())
                pl[j].append(probabilities[j][now_id].item())
                el[j].append(entropy_sum[j].item())

    return outputs_, pl, el, pl_nocate, el_nocate
#------------------------------------------------------------#

#------------------------------------------------------------#
prompts_extract = json.load(open("./prompt/extract_with_entity.json"))
prompts_extract_no_entity = json.load(open("./prompt/extract.json"))
def get_new(sentences, model, llmtokenizer, type_=None):
    msgs = []

    for s, t in zip(sentences, type_):
        if t is None or t.strip() == "":
            msgs.append(prompts_extract_no_entity + [{"role": "user", "content": f"Sentence: {s}"}])
        else:
            msgs.append(prompts_extract+ [{"role": "user", "content": f"Type of the masked entity: {t}.\nSentence: {s}"}])

   
    generation_config = dict(
                            do_sample=False,
                            num_beams=1,
                            max_new_tokens=100,
                            return_dict_in_generate=True,
                            output_hidden_states=True,
                            output_scores = True)
    input_ids = llmtokenizer.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        padding=True,
    )
    # for k in range(len(input_ids)):
    #     print(len(input_ids[k]))
    input_ids = torch.tensor(input_ids).to(model.device)
    outputs = model.generate(
        input_ids,
        pad_token_id=llmtokenizer.eos_token_id,
        attention_mask=input_ids.ne(llmtokenizer.eos_token_id),
        **generation_config
    )
    outputs_ = []
    for i in range(len(sentences)):
        response = outputs.sequences[i][input_ids.shape[-1]:]   
        output_ = llmtokenizer.decode(response, skip_special_tokens=True).split("\n")[0].strip()
        if output_.endswith("."):
            output_ = output_[:-1]
        outputs_.append(output_)
    return outputs_
#------------------------------------------------------------#

determine_prompts = json.load(open("./prompt/determine.json"))
#-------------------------------------------#
def conflict_pd(sentence, fact, model, llmtokenizer):
    # print(f"Sentence: {sentence}")
    # print(f"Fact: {fact}")
    msgs = determine_prompts + [{"role": "user", "content": f"Fact: {fact}\nSentence: {sentence}"}]
    input_ids = llmtokenizer.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        padding=True,
    )
    input_ids = torch.tensor([input_ids]).to(model.device)
    with torch.no_grad():
        op = model(input_ids)

        logit = op.logits
        # probabilities = F.softmax(logit, dim=2)
        yes_token_id = llmtokenizer.encode("yes", add_special_tokens=False)[0]
        no_token_id = llmtokenizer.encode("no", add_special_tokens=False)[0]
        yes_prob = logit[0][-1][yes_token_id].item()
        no_prob = logit[0][-1][no_token_id].item()
        
    # print(f"yes: {yes_prob}, no: {no_prob}, result: {yes_prob > no_prob}")
    return yes_prob > no_prob


match_prompts = json.load(open("./prompt/category.json", "r"))

def match_pd(e, t, model, llmtokenizer):
    # print(f"Sentence: {sentence}")
    # print(f"Fact: {fact}")
    msgs = match_prompts + [{"role": "user", "content": f"Entity: {e}\nAssigned Type: {t}"}]
    input_ids = llmtokenizer.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        padding=True,
    )
    input_ids = torch.tensor([input_ids]).to(model.device)
    with torch.no_grad():
        op = model(input_ids)

        logit = op.logits
        # probabilities = F.softmax(logit, dim=2)
        yes_token_id = llmtokenizer.encode("yes", add_special_tokens=False)[0]
        no_token_id = llmtokenizer.encode("no", add_special_tokens=False)[0]
        yes_token_id2 = llmtokenizer.encode("Yes", add_special_tokens=False)[0]
        no_token_id2 = llmtokenizer.encode("No", add_special_tokens=False)[0]
        yes_prob = logit[0][-1][yes_token_id].item()
        no_prob = logit[0][-1][no_token_id].item()
        yes_prob2 = logit[0][-1][yes_token_id2].item()
        no_prob2 = logit[0][-1][no_token_id2].item()
        
    # print(f"yes: {yes_prob}, no: {no_prob}, result: {yes_prob > no_prob}")
    return yes_prob + yes_prob2 > no_prob + no_prob2