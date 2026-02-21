import random
from tqdm import tqdm

from src.load_data import load_json
from src.gold_extraction import extract_gold, merge_gold_sentences
from src.prompt_builder import (
    build_gold_prompt,
    build_retrieved_only_prompt,
    build_retrieved_random_prompt,
)
from src.inference import load_model, generate
from src.evaluate import is_correct


# -----------------------------
# Config
# -----------------------------

DATA_PATH = "data/full_data.json"
RANDOM_PATH = "data/random_docs.json"

# Try multiple q values
Q_VALUES = [1, 3, 5]

# Total docs in retrieved+random
TOTAL_DOCS = 10

MAX_SAMPLES =  1200 # set to e.g. 200 for quick test


# -----------------------------
# Load Data
# -----------------------------

print("Loading data...")

data = load_json(DATA_PATH)
random_pool = load_json(RANDOM_PATH)

if MAX_SAMPLES:
    data = data[:MAX_SAMPLES]

print("Samples:", len(data))


# -----------------------------
# Load Model
# -----------------------------

print("Loading model...")

model, tokenizer = load_model()


# -----------------------------
# Evaluation Functions
# -----------------------------

def eval_gold(data):
    correct = 0

    for ex in tqdm(data, desc="Gold"):

        gold = merge_gold_sentences(extract_gold(ex))

        prompt = build_gold_prompt(ex["question"], gold)

        pred = generate(model, tokenizer, prompt)

        if is_correct(pred, [ex["answer"]]):
            correct += 1

    return correct / len(data)


def eval_retrieved_only(data, q):
    correct = 0

    for ex in tqdm(data, desc=f"Retrieved-only (q={q})"):

        retrieved = ex["retrieved"][:q]

        prompt = build_retrieved_only_prompt(
            ex["question"], retrieved
        )

        pred = generate(model, tokenizer, prompt)

        if is_correct(pred, [ex["answer"]]):
            correct += 1

    return correct / len(data)


def eval_retrieved_random(data, q):
    correct = 0
    num_random = TOTAL_DOCS - q

    for ex in tqdm(data, desc=f"Retrieved+Random (q={q})"):

        retrieved = ex["retrieved"][:q]

        random_docs = random.sample(
            random_pool, num_random
        )

        prompt = build_retrieved_random_prompt(
            ex["question"],
            retrieved,
            random_docs,
        )

        pred = generate(model, tokenizer, prompt)

        if is_correct(pred, [ex["answer"]]):
            correct += 1

    return correct / len(data)


# -----------------------------
# Run Experiments
# -----------------------------

results = []

print("\nRunning Gold baseline...")
gold_acc = eval_gold(data)

print(f"\nGold accuracy: {gold_acc:.4f}\n")


for q in Q_VALUES:

    print(f"\n===== q = {q} =====")

    ret_only = eval_retrieved_only(data, q)
    ret_rand = eval_retrieved_random(data, q)

    results.append({
        "q": q,
        "gold": gold_acc,
        "retrieved": ret_only,
        "retrieved_random": ret_rand,
    })


# -----------------------------
# Print Results
# -----------------------------

print("\n\n========== FINAL RESULTS ==========\n")

print("| q | Gold | Retrieved | Retrieved+Random |")
print("|---|------|-----------|------------------|")

for r in results:
    print(
        f"| {r['q']} "
        f"| {r['gold']:.4f} "
        f"| {r['retrieved']:.4f} "
        f"| {r['retrieved_random']:.4f} |"
    )

print("\nDone.")
