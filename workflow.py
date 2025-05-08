import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration
import os
import sys

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda if torch.cuda.is_available() else 'N/A')

# === Setup device
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
except Exception as e:
    print(f"Error setting up device: {str(e)}")
    raise

# === Load Emotion Classifier
try:
    emotion_model_path = "./mental_health_bert_model_best"
    print(f"Loading emotion model from: {os.path.abspath(emotion_model_path)}")
    print("Files in emotion model directory:", os.listdir(emotion_model_path))
    
    # First load tokenizer
    emotion_tokenizer = BertTokenizer.from_pretrained(emotion_model_path)
    print("Emotion tokenizer loaded successfully")
    
    # Then load model
    emotion_model = BertForSequenceClassification.from_pretrained(emotion_model_path)
    print("Emotion model loaded successfully")
    
    # Move to device
    emotion_model = emotion_model.to(device)
    print(f"Emotion model moved to {device}")
    
    emotion_model.eval()
    print("Emotion model set to eval mode")
except Exception as e:
    print(f"Error loading emotion model: {str(e)}")
    print(f"Error type: {type(e)}")
    import traceback
    traceback.print_exc()
    raise

# === Load Intent Classifier
try:
    intent_model_path = "./intent_classifier_bert_best"
    print(f"Loading intent model from: {os.path.abspath(intent_model_path)}")
    
    intent_tokenizer = BertTokenizer.from_pretrained(intent_model_path)
    print("Intent tokenizer loaded successfully")
    
    intent_model = BertForSequenceClassification.from_pretrained(intent_model_path)
    print("Intent model loaded successfully")
    
    intent_model = intent_model.to(device)
    print(f"Intent model moved to {device}")
    
    intent_model.eval()
    print("Intent model set to eval mode")
except Exception as e:
    print(f"Error loading intent model: {str(e)}")
    raise

# === Load Coping Tip Generator
try:
    t5_model_path = "./t5_coping_model"
    print(f"Loading T5 model from: {os.path.abspath(t5_model_path)}")
    
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_path)
    print("T5 tokenizer loaded successfully")
    
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_path)
    print("T5 model loaded successfully")
    
    t5_model = t5_model.to(device)
    print(f"T5 model moved to {device}")
    
    t5_model.eval()
    print("T5 model set to eval mode")
except Exception as e:
    print(f"Error loading T5 model: {str(e)}")
    raise

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
