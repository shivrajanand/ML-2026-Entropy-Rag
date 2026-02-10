from src.load_data import load_json
from src.gold_extraction import extract_gold, merge_gold_sentences
from src.prompt_builder import build_gold_prompt
from src.inference import load_model, generate
from src.evaluate import is_correct

data = load_json("data/full_data.json")

model, tokenizer = load_model()

correct = 0

for ex in data:
    gold = merge_gold_sentences(extract_gold(ex))
    prompt = build_gold_prompt(ex["question"], gold)
    pred = generate(model, tokenizer, prompt)

    if is_correct(pred, [ex["answer"]]):
        correct += 1

accuracy = correct / len(data)
print("Gold accuracy:", accuracy)
