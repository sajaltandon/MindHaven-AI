import torch
from transformers import BertTokenizer, BertForSequenceClassification

# === Load Model & Tokenizer ===
model_path = "./mental_health_bert_model_best"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# === Predict Function ===
def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    label = model.config.id2label[predicted_class]
    return label

# === User Input Loop ===
print("\n🧠 Mental Health Classifier - Type 'exit' to quit.\n")

while True:
    user_input = input("📝 You: ")
    if user_input.lower().strip() in ["exit", "quit"]:
        print("👋 Goodbye! Take care.")
        break

    prediction = classify_text(user_input)
    print(f"\n📄 Input: {user_input}")
    print(f"🤖 Prediction: {prediction}")
    print("-" * 50)
