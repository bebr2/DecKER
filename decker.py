import re
from utils import retrieve_facts, get_new, get_mask, match_pd, conflict_pd
import time



def Sample_MASK_COT(q, model, llmtokenizer, num_samples=6):
    assert num_samples > 1
    mask_cots, pls, els, pls_notcate, els_notcate = get_mask(q, model, llmtokenizer, num_return_sequences=num_samples-1)
    filtered_mask_cots = []
    for j, cot in enumerate(mask_cots):
        # print(j)
        # print(cot)
        # print("=====================================")
        if "\n[CATEGORY]" not in cot:
            continue
        if "MASK ANS" not in cot:
            continue
        categories = cot.split("\n[CATEGORY]")[-1].strip()

        categories = re.findall(r'\[([^\]]+)\]', categories)
        categories = [c for c in categories if "MASK" not in c]
        mask_cot = cot.split("\n[CATEGORY]")[0]
        masked_part = re.findall(r'\[MASK [^\]]+\]', cot)
        mask_num_l = 0
        for mask in masked_part:
            try:
                mask_num_l = max(mask_num_l, int(mask[6:-1]))
            except:
                pass
        masked_part = list(set(masked_part))
        masked_part.sort()
        mask_num_l += 1
        cot_data = {
            "idx": j,
            "mask_cot": mask_cot,
            "categories": categories,
            "masked_part": masked_part,
            "masked_num": mask_num_l,
            "cate_pl": pls[j],
            "cate_el": els[j],
            "notcate_pl": pls_notcate[j],
            "notcate_el": els_notcate[j],
        }
            
        mask_map = {}
        for m in range(min(len(masked_part), len(categories))):
            mask_map[masked_part[m]] = categories[m]
        cot_data["mask_map"] = mask_map
        filtered_mask_cots.append(cot_data)
    return filtered_mask_cots

def Fillin_MASK_COT(contriever, tokenizer, model, llmtokenizer, mask_cot_data, embs, n2o, new_facts, target_facts, diff_bound, abs_bound):
    mask_cot = mask_cot_data["mask_cot"]
    types = mask_cot_data["categories"]
    mask_parts = mask_cot_data["masked_part"]
    mask_map = {}
    for i in range(min(len(mask_parts), len(types))):
        mask_map[mask_parts[i]] = types[i]
    cots = [c.strip() for c in mask_cot.split("[STEP]") if c.strip()]
    now_cot = []
    now_keywords = [""]
    used_new_facts = []
    ans = ""
    docs = []
    for i in range(len(cots)):
        masked_part = re.findall(r'\[MASK [^\]]+\]', cots[i])
        if not masked_part:
            now_keywords.append("")
            now_cot.append(cots[i])
            continue   
        qr = cots[i]
        for m in masked_part:
            qr = qr.replace(m, " ")
        query = (now_keywords[i] + "\n" + qr).strip()
        
        if len(new_facts) > 1:
            fact_ids, fact_value = retrieve_facts(query, embs, contriever, tokenizer)
            fact_id = fact_ids[0]
            # new_fact = o2n.get(all_facts[fact_id])
            new_fact = new_facts[fact_id]
            old_fact = n2o[new_fact]
            
            conflict1 = fact_value[0] - fact_value[1] > diff_bound
            conflict2 = fact_value[0] > abs_bound
            if conflict1 == conflict2:
                conflict = conflict1
            else:
                conflict = conflict_pd(cots[i], old_fact, model, llmtokenizer)
        else:
            fact_ids, fact_value = retrieve_facts(query, embs, contriever, tokenizer, k=1)
            fact_id = fact_ids[0]
            # new_fact = o2n.get(all_facts[fact_id])
            new_fact = new_facts[fact_id]
            old_fact = n2o[new_fact]
            
            conflict = conflict_pd(cots[i], old_fact, model, llmtokenizer)
        
        
        is_conflict = (fact_id not in used_new_facts) and conflict

        

        docs.append([query, n2o[new_fact], new_fact, fact_value, is_conflict])
        
        if is_conflict:
            mask_word = target_facts[new_fact]
            used_new_facts.append(fact_id)
        else:
            mask_word = yield True, (cots[i], mask_map[masked_part[0]] if masked_part[0] in mask_map else " ")
        
        for l in range(i, len(cots)):
            cots[l] = cots[l].replace(masked_part[0], mask_word)
        if masked_part[0] == "[MASK ANS]":
            ans = mask_word
        now_keywords.append(mask_word)
        now_cot.append(cots[i])
        
    yield False, {
        "new_thoughts": now_cot,
        "new_answer": ans,
        "docs": docs,
        "masked_words": [nk for nk in now_keywords[1:] if nk]
    }
    
def Score_COT(model, llmtokenizer, categories, masked_words):
    results = []
    # categories
    for category_list, masked_word_list in zip(categories, masked_words):
        results.append([])
        for i in range(min(len(masked_word_list), len(category_list))):
            right = match_pd(masked_word_list[i], category_list[i], model, llmtokenizer)
            results[-1].append({
                "entity": masked_word_list[i],
                "category": category_list[i],
                "right": right,
            })
    return results

def get_Fillin_output(all_mask_cot_data, model, llmtokenizer, contriever, tokenizer, mask_cot_data, embs, n2o, new_facts, target_new_facts, diff_bound, abs_bound):
    generators = []
    for mask_cot_data in all_mask_cot_data:
        generators.append(Fillin_MASK_COT(contriever, tokenizer, model, llmtokenizer, mask_cot_data, embs, n2o, new_facts, target_new_facts, diff_bound, abs_bound))
    results = [None] * len(generators)
    batch = [None] * len(generators)
    completed = [False] * len(generators)
    while not all(completed):
        for i, g in enumerate(generators):
            if completed[i]:
                continue
            try:
                if batch[i] is not None:
                    is_llm, text = g.send(batch[i])
                    batch[i] = None
                else:
                    is_llm, text = next(g)
                if is_llm:
                    batch[i] = text
                else:
                    results[i] = text
                    completed[i] = True
            except StopIteration as e:
                completed[i] = True
                continue
        if any(batch):
            batch_sentence = [text[0] for text in batch if text is not None]
            batch_types = [text[1] for text in batch if text is not None]
            llm_results = get_new(batch_sentence, model, llmtokenizer, batch_types)

            for i, g in enumerate(generators):
                if batch[i] is not None:
                    batch[i] = llm_results.pop(0)
    reses = []
    for j in range(len(all_mask_cot_data)):
        reses.append({
            "new": results[j],
            "mask_cot": all_mask_cot_data[j],
        })
    return reses

def DecKER(q, model, llmtokenizer, contriever, tokenizer, embs, n2o, new_facts, target_new_facts, diff_bound, abs_bound, num_samples=6):
    reses = []
    all_mask_cot_data = Sample_MASK_COT(q, model, llmtokenizer, num_samples=num_samples)
    
    if not all_mask_cot_data:
        return None
    reses = get_Fillin_output(all_mask_cot_data, model, llmtokenizer, contriever, tokenizer, all_mask_cot_data, embs, n2o, new_facts, target_new_facts, diff_bound
                              , abs_bound)
    cate_scores = Score_COT(model, llmtokenizer, [r["categories"] for r in all_mask_cot_data], [r["new"]["masked_words"] for r in reses])
    return reses, cate_scores