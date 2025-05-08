# Mental Health NLP Project

This project implements a comprehensive Natural Language Processing (NLP) system for mental health support, consisting of three main components:

1. **T5 Coping Strategy Generator**: Generates coping strategies based on user input
2. **BERT Intent Classifier**: Classifies user intents from their messages
3. **BERT Mental Health Classifier**: Analyzes mental health-related content

## Project Structure

```
├── model1.ipynb                 # Development notebook for model 1
├── model2.ipynb                 # Development notebook for model 2
├── model3.ipynb                 # Development notebook for model 3
├── inference.py                 # Inference script for model 1
├── inference2.py               # Inference script for model 2
├── inference3.py               # Inference script for model 3
├── workflow.py                 # Main workflow script
├── datasets/
│   ├── T5_Coping_Tip_Dataset__Expanded_.csv
│   ├── Intent_Classification_Dataset__5000_entries_.csv
│   └── Mental_Health_Dataset__4000_entries_.csv
└── models/
    ├── t5_coping_model/
    ├── intent_classifier_bert_best/
    └── mental_health_bert_model_best/
```

## Setup and Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. For inference using the models:
```bash
python inference.py
python inference2.py
python inference3.py
```

2. For the complete workflow:
```bash
python workflow.py
```

## Model Details

### T5 Coping Strategy Generator
- Based on T5 architecture
- Generates coping strategies based on user input
- Trained on expanded coping tip dataset

### BERT Intent Classifier
- Fine-tuned BERT model
- Classifies user intents
- Trained on 5000 entries dataset

### BERT Mental Health Classifier
- Fine-tuned BERT model
- Analyzes mental health content
- Trained on 4000 entries dataset

## Note
The model weights and large files are not included in this repository due to size limitations. Please contact the maintainers for access to the model files.

## License
[Your chosen license] 