# MindHaven AI - Mental Health Support System

MindHaven AI is an advanced Natural Language Processing (NLP) system designed to provide mental health support through intelligent conversation and coping strategy generation. The system combines multiple state-of-the-art language models to understand user intents, analyze mental health content, and generate personalized coping strategies.

## 🌟 Features

- **Intelligent Intent Classification**: Understands user messages and classifies their intents using BERT
- **Mental Health Analysis**: Analyzes mental health-related content using specialized  fine-tuned BERT model
- **Personalized Coping Strategies**: Generates contextually relevant coping strategies using T5 model
- **Comprehensive Workflow**: Integrated system that combines all components for seamless interaction

## 🏗️ Architecture

The system consists of three main components:

1. **T5 Coping Strategy Generator**
   - Based on T5 architecture
   - Generates personalized coping strategies
   - Trained on expanded dataset of coping tips
   - Handles various mental health scenarios

2. **BERT Intent Classifier**
   - Fine-tuned BERT model
   - Classifies user intents from messages
   - Trained on 5000+ entries
   - Supports multiple intent categories

3. **BERT Mental Health Classifier**
   - Specialized BERT model
   - Analyzes mental health content
   - Trained on 4000+ entries
   - Provides mental health context analysis

## 📦 Project Structure

```
├── model1.ipynb                 # Development notebook for T5 model
├── model2.ipynb                 # Development notebook for Intent Classifier
├── model3.ipynb                 # Development notebook for Mental Health Classifier
├── inference.py                 # Inference script for T5 model
├── inference2.py               # Inference script for Intent Classifier
├── inference3.py               # Inference script for Mental Health Classifier
├── workflow.py                 # Main workflow integration script
├── datasets/
│   ├── T5_Coping_Tip_Dataset__Expanded_.csv
│   ├── Intent_Classification_Dataset__5000_entries_.csv
│   └── Mental_Health_Dataset__4000_entries_.csv
└── models/
    ├── t5_coping_model/
    ├── intent_classifier_bert_best/
    └── mental_health_bert_model_best/
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git LFS (for handling large model files)
- Sufficient disk space for model files

### Installation

1. Clone the repository:
```bash
git clone https://github.com/sajaltandon/MindHaven-AI.git
cd MindHaven-AI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

1. For individual model inference:
```bash
# Generate coping strategies
python inference.py

# Classify user intent
python inference2.py

# Analyze mental health content
python inference3.py
```

2. For complete workflow:
```bash
python workflow.py
```

## 📊 Model Performance

- **T5 Coping Strategy Generator**: Generates contextually relevant coping strategies
- **BERT Intent Classifier**: High accuracy in intent classification
- **BERT Mental Health Classifier**: Precise mental health content analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Sajal Tandon** - *Initial work* - [GitHub Profile](https://github.com/sajaltandon)

## 🙏 Acknowledgments

- Thanks to the open-source community for the amazing tools and libraries
- Special thanks to Hugging Face for their transformer models and tools
- Gratitude to all contributors and supporters of this project

## ⚠️ Important Note

This system is designed to provide support and coping strategies but is not a replacement for professional mental health care. Always consult with qualified mental health professionals for serious concerns. 
