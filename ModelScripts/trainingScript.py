import os
import pandas as pd
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer

# Check for CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load dataset from CSV
def load_shakespeare_dataset(file_path="shakespeare_dataset.csv"):
    df = pd.read_csv(file_path)
    # Group by stanza/section to create complete texts
    grouped = df.groupby(['File', 'Genre']).agg({'line': lambda x: ' '.join(x)}).reset_index()
    
    # Create text samples in the format needed for training
    dataset = Dataset.from_pandas(grouped)
    
    # Format data for training
    def format_data(example):
        return {
            "text": example["line"],
            "genre": example["Genre"]
        }
    
    dataset = dataset.map(format_data)
    return dataset

# Set up model parameters
max_seq_length = 512  # Reduced for 4060 memory constraints
lora_rank = 48 # Adjusted for performance/memory tradeoff
model_name = "meta-llama/meta-Llama-3.1-8B"  # Non-instruct model for raw text training

# Load the model with 4-bit quantization optimized for your 4060
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,  # Use 4-bit quantization for your 4060
    fast_inference=False,  # Disable vLLM as it requires more memory
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.85,  # Adjust based on your 4060's VRAM
)

# Set up LoRA for efficient training
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=lora_rank*2,
    use_gradient_checkpointing="unsloth",  # Enable efficient training
    random_state=42,
)

# Load and prepare the Shakespeare dataset
dataset = load_shakespeare_dataset()
print(f"Dataset size: {len(dataset)} samples")

# Function to format the data for continuous text training
def formatting_func(example):
    # For plain text training without instruction format
    return example["text"]

# Set up training arguments with conservative parameters for your 4060
training_args = TrainingArguments(
    output_dir="shakespeare_model",
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Small batch size for 4060
    gradient_accumulation_steps=4,  # Increase to compensate for small batch size
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    learning_rate=2e-5,
    weight_decay=1.0,
    fp16=False,  # Disable mixed precision since we're using 4-bit
    bf16=False,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    logging_steps=5,
    save_steps=100,
    save_total_limit=2,
    report_to="none",  # Disable W&B reporting
)

# Configure the SFTTrainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    args=training_args,
    packing=True,  # Enable packing for more efficient training
    formatting_func=formatting_func,
    max_seq_length=max_seq_length,
)

# Start training
print("Starting training...")
trainer.train()

# Save the trained model
output_dir = "modelm0.3.2"
os.makedirs(output_dir, exist_ok=True)


# Code for inference with the trained model
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

# Test the model with a sample prompt
if __name__ == "__main__":
    print("\nTesting the trained model:")
    test_prompt = "Shall I compare thee to"
    result = generate_shakespeare_text(test_prompt)
    print(f"Prompt: {test_prompt}")
    print(f"Generated text:\n{result}")

# Save LoRA weights
model.save_pretrained(output_dir)
#model.save_pretrained_merged(output_dir, tokenizer, save_method = "merged_8bit",)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")
#Optionally convert to GGUF format for use with llama.cpp
#print("\nTo convert to GGUF for llama.cpp, uncomment and run the following code:")
#model.save_pretrained_gguf("shakespeare_gguf", tokenizer, quantization_method="q4_k_m")
