INSTRUCTION = (
    "You are given a question and a set of documents.\n"
    "Answer the question using only the information in the documents.\n"
    "If the answer cannot be found, respond with NO-RES.\n"
    "Answer concisely."
)


def normalize_doc(doc, default_title):
    if isinstance(doc, str):
        return {
            "title": default_title,
            "text": doc
        }
    elif isinstance(doc, dict):
        return {
            "title": doc.get("title", default_title),
            "text": doc.get("text", "")
        }
    else:
        raise TypeError(f"Unsupported doc type: {type(doc)}")


def format_documents(docs):
    blocks = []
    for i, doc in enumerate(docs, 1):
        blocks.append(
            f"Document [{i}]:\n{doc['title']}\n{doc['text']}"
        )
    return "\n\n".join(blocks)


def build_gold_prompt(question, gold_docs):
    return (
        f"{INSTRUCTION}\n\n"
        f"Documents:\n"
        f"{format_documents(gold_docs)}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


def build_retrieved_random_prompt(question, retrieved_docs, random_docs):

    docs = []

    # Random first (noise)
    for d in random_docs:
        docs.append(normalize_doc(d, "Random"))

    # Retrieved last (signal)
    for d in retrieved_docs:
        docs.append(normalize_doc(d, "Retrieved"))

    return (
        f"{INSTRUCTION}\n\n"
        f"Documents:\n"
        f"{format_documents(docs)}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


def build_retrieved_only_prompt(question, retrieved_docs):

    docs = []

    for d in retrieved_docs:
        docs.append(normalize_doc(d, "Retrieved"))

    return (
        f"{INSTRUCTION}\n\n"
        f"Documents:\n"
        f"{format_documents(docs)}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
