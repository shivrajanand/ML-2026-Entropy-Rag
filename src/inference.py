import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer

def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.1-8B-Instruct",
        max_seq_length=4096,
        load_in_4bit=True,
        dtype=None
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    output = model.generate(
        **inputs,
        max_new_tokens=15,
        do_sample=False
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.split("Answer:")[-1].strip()
