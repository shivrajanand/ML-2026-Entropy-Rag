def extract_gold(entry):
    context = entry["context"]
    supporting = entry["supporting_facts"]

    title_to_sents = {title: sents for title, sents in context}

    gold = []
    for title, sent_id in supporting:
        if title in title_to_sents:
            sents = title_to_sents[title]
            if 0 <= sent_id < len(sents):
                gold.append({
                    "title": title,
                    "text": sents[sent_id]
                })
    return gold


def merge_gold_sentences(gold_list):
    merged = {}
    for g in gold_list:
        merged.setdefault(g["title"], []).append(g["text"])

    docs = []
    for title, sents in merged.items():
        docs.append({
            "title": title,
            "text": " ".join(sents)
        })
    return docs
