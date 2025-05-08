import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# === Load trained model and tokenizer ===
model_path = r"C:\Users\sajal\Desktop\NLP Project_updated\t5_coping_model"  # (or "./t5_coping_model/checkpoint-xxx" if needed)
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# === Function to generate coping tip ===
def generate_coping_tip(emotion, intent):
    prompt = f"generate coping tip: emotion={emotion} | intent={intent}"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=64,
            num_beams=4,
            early_stopping=True
        )

    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded_output

# === CLI Interaction ===
print("\n🧠 Coping Tip Generator Ready! Type 'exit' to quit.\n")

while True:
    emotion = input("Enter emotion (anxiety / depression / stress / neutral): ").strip().lower()
    if emotion == "exit":
        break
    intent = input("Enter intent (venting / seeking_support / reflecting / sharing_update / asking_question): ").strip().lower()
    if intent == "exit":
        break

    tip = generate_coping_tip(emotion, intent)
    print(f"\n📄 Prompt: emotion={emotion} | intent={intent}")
    print(f"💡 Coping Tip: {tip}")
    print("-" * 50)
