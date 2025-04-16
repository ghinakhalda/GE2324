from unsloth import FastLanguageModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

if __name__ == "__main__":
    print("\nTesting the trained model:")
    for i in shakespearean_phrases:
        result = generate_shakespeare_text(i)
        print(result)
        print("/////////////////////////")
