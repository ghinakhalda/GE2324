from unsloth import FastLanguageModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import numpy as np
import pandas as pd
from nltk import ngrams
from collections import Counter

device = "cuda"
model, tokenizer = FastLanguageModel.from_pretrained(
    "Lora_model-0.3.2/modelm0.3.2",
    device_map={"": device}  
)

def generate_shakespeare_text(prompt, max_length=200):
    # Format the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = inputs["input_ids"].size(1)
    # Generate with LoRA
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
            output_scores=True,
            return_dict_in_generate=True,
        )
    
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    return generated_text, outputs, prompt_length

def compute_perplexity_from_outputs(outputs, prompt_length):
    scores = outputs.scores 
    sequences = outputs.sequences[0, prompt_length:]  
    nll = 0
    num_tokens = len(sequences)
    
    if num_tokens == 0:
        return float("inf")
    
    for i in range(num_tokens):
        log_probs = torch.log_softmax(scores[i], dim=-1)
        token_id = sequences[i].item() 
        nll -= log_probs[0, token_id].item()
    
    return np.exp(nll / num_tokens)


def get_trigrams(text):
    words = text.split()
    return [' '.join(gram) for gram in ngrams(words, 3)] if len(words) >= 3 else []

def compute_plagiarism_percentage_from_outputs(outputs, training_data, column='line'):
    # Get output trigrams from generated text and training data
    output_trigrams = get_trigrams(outputs.lower())
    train_data_trigrams = set()
    for text in training_data[column].dropna().str.lower():
        train_data_trigrams.update(get_trigrams(text))
    
    # Find and count matched trigrams
    output_trigrams_count = Counter(output_trigrams)
    total_trigrams = sum(output_trigrams_count.values())
    plagiarized_trigrams = sum(count for trigram, count in output_trigrams_count.items() if trigram in train_data_trigrams)
    
    plagiarism_percentage = (plagiarized_trigrams / total_trigrams) * 100
    
    return plagiarism_percentage
    
    
shakespearean_phrases = [
    "Thou art more fair than dawn",
    "Shall I speak of thee now",
    "Dost thy heart not tremble",
    "Wouldst thou linger in dream",
    "Speak not of sorrow again",
    "Let not time steal thee",
    "Wherefore dost thou retreat so",
    "Mine eyes have seen thy grace",
    "Love's labour is not lost",
    "Bid me rise with thee",
    "Hast thou known truer bliss",
    "I do beseech thee stay",
    "Thine eyes are stars above",
    "Mark me well, gentle soul",
    "Let thy silence speak loud",
    "To thine own self be",
    "When roses bloom, I wait",
    "Swear not by the moon",
    "My heart is thine still",
    "Tread softly on my thoughts",
    "I drink to thee nightly",
    "Thy name doth echo deep",
    "Soft words fall from thee",
    "Wilt thou walk with me",
    "The night doth call sweetly",
    "Give me thy hand fast",
    "I shall not forget thee",
    "Come hither, sweetened by time",
    "Mine honor lies with thee",
    "Look how the light dances",
    "Stars do envy thine eyes",
    "Say thou wilt remember me",
    "Time flies but love lingers",
    "Oh speak again, bright angel",
    "This hour is ours alone",
    "Fear not what fate brings",
    "Thy voice rings like bells",
    "Love doth guide us still",
    "My soul clings to thine",
    "Hold fast to fleeting joy",
    "Leave me not in silence",
    "This day shall not end",
    "Thy presence makes days bright",
    "Thou art my soul’s fire",
    "Words do falter near thee",
    "In dreams I hold thee",
    "Let the heavens see us",
    "The moon sings for thee",
    "Forever is but one kiss",
    "Hearts do dance when near",
    "Let us part nevermore"
]

df = pd.DataFrame({'line': shakespearean_phrases})
df['generated_text'] = ""
df['perplexity'] = float("inf")
df['plagiarized_trigrams'] = float("inf")

if __name__ == "__main__":
    train_data = pd.read_csv('shakespeare_preprocess.csv')
    print("\nTesting the trained model:")
    
    for i in shakespearean_phrases:
        gen_text, outputs, prompt_length = generate_shakespeare_text(i)
        df.loc[df['line']==i, 'generated_text'] = gen_text
        print(gen_text)
        
        perplexity = compute_perplexity_from_outputs(outputs, prompt_length)
        plagiarized_trigrams = compute_plagiarism_percentage_from_outputs(gen_text, train_data)

        df.loc[df['line']==i, 'perplexity'] = perplexity
        df.loc[df['line']==i, 'plagiarized_trigrams'] = plagiarized_trigrams
        
        print("Evaluation Metrics:")
        print(f"Perplexity: {perplexity:.4f}")
        print(f"Percentage of Plagiarized Trigrams: {plagiarized_trigrams:.4f}")
        print()
        print("/////////////////////////")        
        
df.to_csv('test_result.csv', index=False) 