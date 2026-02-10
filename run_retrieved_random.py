import random
from src.load_data import load_json
from src.prompt_builder import build_retrieved_random_prompt
from src.inference import load_model, generate
from src.evaluate import is_correct

data = load_json("data/full_data.json")
random_docs_pool = load_json("data/random_docs.json")

model, tokenizer = load_model()

correct = 0

for ex in data:
    retrieved = ex["retrieved"][:3]   # q = 3
    random_docs = random.sample(random_docs_pool, 7)

    prompt = build_retrieved_random_prompt(
        ex["question"], retrieved, random_docs
    )

    pred = generate(model, tokenizer, prompt)

    if is_correct(pred, [ex["answer"]]):
        correct += 1

accuracy = correct / len(data)
print("Retrieved + Random accuracy:", accuracy)
