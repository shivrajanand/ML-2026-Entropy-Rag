# RAG Evaluation with Llama-3.1-8B-Instruct

This project evaluates the performance of a Large Language Model (LLM) in a Retrieval-Augmented Generation (RAG) setting, inspired by the paper **"The Power of Noise"**.

We compare how the model answers questions under two conditions:

1. **Gold Setting**: The model is given only documents that contain the correct answer.
2. **Retrieved + Random Setting**: The model is given retrieved documents plus random unrelated documents (noise).

The goal is to measure how these different contexts affect answer accuracy.

---

## Table of Contents

- Project Motivation  
- Dataset Description  
- Project Structure  
- Environment Setup  
- Methodology  
  - Gold Document Extraction  
  - Prompt Construction  
  - Model Inference  
  - Accuracy Evaluation  
- Running the Experiments  
- Results  
- Design Choices  
- References  

---

## Project Motivation

Large Language Models sometimes generate incorrect answers due to limited factual grounding.

Retrieval-Augmented Generation (RAG) improves this by providing external documents to the model during inference.

However, retrieved documents can be noisy. Recent research shows that adding random documents may improve model robustness.

This project reproduces and studies that behavior using Llama-3.1-8B-Instruct.

---

## Dataset Description

The project uses two datasets.

### full_data.json

Contains question-answer samples with the following fields:

- _id: Unique identifier  
- question: Input question  
- answer: Ground truth answer  
- context: Wikipedia passages (structured)  
- supporting_facts: Indices of answer-supporting sentences  
- retrieved: Retrieved passages (list of strings)  

### random_docs.json

Contains random Wikipedia passages used as noise.  
Each entry is a plain text string.

---

## Project Structure

project/  
├── data/  
│   ├── full_data.json  
│   ├── random_docs.json  
│  
├── src/  
│   ├── load_data.py  
│   ├── gold_extraction.py  
│   ├── prompt_builder.py  
│   ├── inference.py  
│   ├── evaluate.py  
│  
├── run_gold.py  
├── run_retrieved_random.py  
└── README.md  

---

## Environment Setup

The project is designed for GPU environments such as A100 or A40.

### Install Dependencies (CUDA 12.1+)

pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -U transformers unsloth accelerate bitsandbytes sentencepiece tqdm numpy scipy scikit-learn

Verify GPU:

import torch  
print(torch.cuda.is_available())

---

## Methodology

### Gold Document Extraction

Each sample contains supporting_facts that specify which sentences in the context support the answer.

Steps:

1. Map Wikipedia titles to sentence lists  
2. Select sentences using supporting_facts  
3. Merge sentences per title  
4. Form gold documents  

These gold documents are guaranteed to contain the correct answer.

---

### Prompt Construction

Each prompt has three components:

1. Instruction  
2. Documents  
3. Question  

Instruction:

You are given a question and a set of documents.  
Answer the question using only the information in the documents.  
If the answer cannot be found, respond with NO-RES.  
Answer concisely.

Document format:

Document [i]:  
title  
text  

Question format:

Question: <question>  
Answer:

---

### Gold Prompt

Uses only gold documents.

[I, ⋆, Q]

---

### Retrieved + Random Prompt

Uses:

- 3 retrieved documents  
- 7 random documents  

Order:

Random documents → Retrieved documents → Question

[I, ▢, q, Q]

---

### Model Inference

Model:

- unsloth/Meta-Llama-3.1-8B-Instruct  
- 4-bit quantization  

Inference settings:

- Greedy decoding  
- max_new_tokens = 15  
- do_sample = False  

Unsloth enables fast inference and reduced memory usage.

---

### Accuracy Evaluation

A prediction is considered correct if the ground truth answer appears as a substring in the model output (case-insensitive).

Accuracy = correct_predictions / total_samples

---

## Running the Experiments

### Gold Setting

python run_gold.py

Output:

Gold accuracy: X.XX

---

### Retrieved + Random Setting

python run_retrieved_random.py

Output:

Retrieved + Random accuracy: Y.YY

---

## Results

| Setting            | Accuracy |
|--------------------|----------|
| Gold Only          | X.XX     |
| Retrieved + Random | Y.YY     |

Gold accuracy is expected to be higher.

---

## Design Choices

- Supporting facts are used to extract gold documents reliably.
- Random documents simulate unstructured noise.
- Greedy decoding ensures deterministic outputs.
- Substring matching follows the evaluation protocol of the paper.

---

## References

1. Cuconasu et al. (2024). The Power of Noise: Redefining Retrieval for RAG Systems. SIGIR.  
2. Unsloth: https://github.com/unslothai/unsloth  
3. HuggingFace Transformers: https://huggingface.co/transformers  

---

## Conclusion

This project evaluates a RAG system under controlled experimental conditions.

By comparing gold-only and retrieved + random prompts, we study how contextual noise affects LLM accuracy.
