from src.load_data import load_json
from src.prompt_builder import build_retrieved_only_prompt
from src.inference import load_model, generate
from src.evaluate import is_correct


data = load_json("data/full_data.json")

model, tokenizer = load_model()

correct = 0


for ex in data:

    # Use same number as before (q = 3)
    retrieved = ex["retrieved"][:3]

    prompt = build_retrieved_only_prompt(
        ex["question"], retrieved
    )

    pred = generate(model, tokenizer, prompt)

    if is_correct(pred, [ex["answer"]]):
        correct += 1


accuracy = correct / len(data)

print("Retrieved-only accuracy:", accuracy)
