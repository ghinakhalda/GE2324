from unsloth import FastLanguageModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model, tokenizer = FastLanguageModel.from_pretrained("modelm0.3.2")

def generate_shakespeare_text(prompt, max_length=200):
    # Format the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Generate with LoRA
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text



if __name__ == "__main__":
    with open("input.txt", 'r') as input_file:
        lines = input_file.readlines()
    processed = []
    for l in lines:
        if len(l) > 5:
            processed.append(generate_shakespeare_text(l.rstrip('\n')))
    with open("output.txt", 'a') as output_file:
        for l in processed:
            output_file.write(l + '\n')


