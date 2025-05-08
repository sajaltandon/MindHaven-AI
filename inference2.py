import torch
from transformers import BertTokenizer, BertForSequenceClassification

# === Load fine-tuned model and tokenizer ===
model_path = "./intent_classifier_bert_best"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# === Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# === Predict intent
def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class_id = torch.argmax(outputs.logits, dim=1).item()

    intent = model.config.id2label[predicted_class_id]
    return intent

# === CLI Interface
print("\n🧠 Intent Classifier Ready — Type 'exit' to quit.\n")

while True:
    user_input = input("📝 You: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        print("👋 Exiting. Stay safe.")
        break

    prediction = predict_intent(user_input)
    print(f"📄 Input: {user_input}")
    print(f"🎯 Predicted Intent: {prediction}")
    print("-" * 50)
