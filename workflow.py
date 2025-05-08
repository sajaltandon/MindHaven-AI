import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration
import os

# === Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Emotion Classifier
emotion_model_path = "./mental_health_bert_model_best"
emotion_tokenizer = BertTokenizer.from_pretrained(emotion_model_path)
emotion_model = BertForSequenceClassification.from_pretrained(emotion_model_path).to(device)
emotion_model.eval()

# === Load Intent Classifier
intent_model_path = "./intent_classifier_bert_best"
intent_tokenizer = BertTokenizer.from_pretrained(intent_model_path)
intent_model = BertForSequenceClassification.from_pretrained(intent_model_path).to(device)
intent_model.eval()

# === Load Coping Tip Generator
t5_model_path = "./t5_coping_model"
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_path)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_path).to(device)
t5_model.eval()

# === Check if model files exist (for debugging on Streamlit Cloud) ===
def check_model_files(model_dir, required_files):
    missing = []
    for f in required_files:
        if not os.path.exists(os.path.join(model_dir, f)):
            missing.append(f)
    if missing:
        print(f"WARNING: Missing files in {model_dir}: {missing}")
    else:
        print(f"All required files found in {model_dir}.")

# Check T5 model files
check_model_files(
    "./t5_coping_model",
    ["config.json", "tokenizer_config.json", "special_tokens_map.json", "spiece.model", "model.safetensors"]
)
# Check Emotion model files
check_model_files(
    "./mental_health_bert_model_best",
    ["config.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "model.safetensors"]
)
# Check Intent model files
check_model_files(
    "./intent_classifier_bert_best",
    ["config.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "model.safetensors"]
)

# === Functions
def classify_emotion(text):
    inputs = emotion_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
        pred_id = torch.argmax(logits, dim=1).item()
    return emotion_model.config.id2label[pred_id]

def predict_intent(text):
    inputs = intent_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        logits = intent_model(**inputs).logits
        pred_id = torch.argmax(logits, dim=1).item()
    return intent_model.config.id2label[pred_id]

def generate_coping_tip(emotion, intent):
    prompt = f"generate coping tip: emotion={emotion} | intent={intent}"
    inputs = t5_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = t5_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=64,
            num_beams=4,
            early_stopping=True
        )
    decoded_output = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output

# === Streamlit UI ===
st.set_page_config(page_title="Mental Health Companion", page_icon="🧠", layout="centered")
st.title("🧠 Mental Health Companion Assistant")

st.markdown("Type your feelings below. The assistant will detect your emotion, understand your intent, and suggest a helpful coping tip.")

user_input = st.text_area("📝 Enter your thoughts here:", height=150)

if st.button("Get Support 💬"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        with st.spinner('Analyzing...'):
            try:
                emotion = classify_emotion(user_input)
                intent = predict_intent(user_input)
                coping_tip = generate_coping_tip(emotion, intent)

                st.success("✅ Analysis Complete!")
                st.write(f"🔍 **Predicted Emotion:** `{emotion}`")
                st.write(f"🎯 **Predicted Intent:** `{intent}`")
                st.write(f"💡 **Coping Tip:** {coping_tip}")

            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
