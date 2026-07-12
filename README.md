# Context-Aware Hate Speech Detection using Transformer Models

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://alwinpaul111-hate-speech-detector-app.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/alwinn/hate-speech-distilbert)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A production ready hate speech detection system built with DistilBERT, featuring sarcasm-aware context understanding, SHAP/LIME explainability, multilingual support (English + Hindi), and a deployed Streamlit web application.

---

##  Live Demo

** [Try the app here](https://hate-speech-detector-ewoeuogd9zxgehywempuje.streamlit.app/)**



---

## Screenshots

| Input | Prediction | SHAP Explanation |
|---|---|---|
| "All immigrants should go back" |  Hate Speech (99.9%) | Words highlighted in red |
| "I love how diverse our community is" | ✅ No Hate (99.3%) | Words highlighted in blue |
| "तुम बहुत घटिया इंसान हो" | 🚨 Hate Speech (99.5%) | Auto-translated + classified |

---

##  Features

| Feature | Description |
|---|---|
|  **DistilBERT Fine-tuning** | Transformer model fine-tuned on balanced hate speech dataset |
|  **Context Understanding** | Sarcasm detection pipeline flags indirect/implicit hate |
|  **SHAP Explainability** | Word-level attribution showing WHY model predicted hate |
|  **LIME Explainability** | Alternative feature importance visualization |
|  **Multilingual** | English + Hindi support via Helsinki-NLP translation |
|  **Deployed Web App** | Live Streamlit app with real-time predictions |

---

##  Architecture

```
Input Text (EN/HI)
       │
       ▼
Language Detection (langdetect)
       │
  Hindi? ──YES──► Helsinki-NLP Translation (HI→EN)
       │                    │
       NO◄──────────────────┘
       │
       ▼
Text Preprocessing (clean URLs, mentions)
       │
       ▼
DistilBERT Tokenizer (max_length=128)
       │
       ▼
Fine-tuned DistilBERT Classifier
       │
       ▼
┌──────┴──────┐
│             │
▼             ▼
Hate       No Hate
Speech         │
│         Sarcasm Check
│              │
│         Sarcastic? ──YES──► ⚠️ Indirect Hate
│              │
│              NO
│              │
▼              ▼
SHAP / LIME Explanation
       │
       ▼
Streamlit Web App Output
```

---

##  Tech Stack

| Category | Tools |
|---|---|
| **Model** | DistilBERT (HuggingFace Transformers) |
| **Fine-tuning** | HuggingFace Trainer API |
| **Explainability** | SHAP, LIME |
| **Translation** | Helsinki-NLP/opus-mt-hi-en |
| **Sarcasm** | mrm8488/t5-base-finetuned-sarcasm-twitter |
| **Web App** | Streamlit |
| **Deployment** | Streamlit Cloud |
| **Model Hosting** | HuggingFace Hub |
| **Language** | Python 3.10+ |

---

##  Project Structure

```
HATE-SPEECH-DETECTOR/
│
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── LICENSE                 # MIT License
│
└── notebooks/
    └── hate_speech_detection.ipynb   # Full training pipeline
```

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/alwinpaul111/HATE-SPEECH-DETECTOR.git
cd HATE-SPEECH-DETECTOR
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app locally
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

##  Model Details

| Parameter | Value |
|---|---|
| Base Model | `distilbert-base-uncased` |
| Dataset | hate_speech_dataset_2000 (balanced, 2025 samples) |
| Train/Val/Test Split | 70% / 15% / 15% |
| Training Epochs | 3 |
| Batch Size | 16 |
| Max Sequence Length | 128 tokens |
| Optimizer | AdamW (weight_decay=0.01) |
| Labels | 0 = No Hate, 1 = Hate Speech |

### Dataset Label Distribution
```
No Hate     : 1,010 samples (50%)
Hate Speech :   1,015 samples (50%)
```
Perfectly balanced dataset — no class imbalance issues.

---

##  Results

| Metric | Score |
|---|---|
| Accuracy | ~92% |
| F1 Score (weighted) | ~0.91 |
| Hate Speech Precision | ~0.93 |
| Hate Speech Recall | ~0.91 |

---

##  Explainability Examples

### SHAP Word-Level Attribution
-  **Red words** push prediction toward **Hate Speech**
-  **Blue words** push prediction toward **No Hate**

Example: *"All immigrants should go back to their country"*
- Words like `immigrants`, `back`, `country` highlighted in red
- Model correctly identifies xenophobic language patterns

### LIME Feature Importance
- Shows top 10 words contributing to the classification
- Weight > 0 → pushes toward Hate Speech
- Weight < 0 → pushes toward No Hate

---

##  Multilingual Support

The app supports **Hindi** input via translation pipeline:

```
Input (Hindi) → langdetect → Helsinki-NLP MT → DistilBERT → Prediction
```

Example:
```
Input     : "तुम बहुत घटिया इंसान हो!"
Translated: "You are a very mean person!"
Result    : 🚨 Hate Speech (99.5% confidence)
```

---

##  How to Train Your Own Model

Open the notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alwinpaul111/HATE-SPEECH-DETECTOR/blob/main/notebooks/hate_speech_detection.ipynb)

The notebook covers:
1. Dataset loading and label fixing
2. Text preprocessing
3. DistilBERT tokenization
4. Fine-tuning with HuggingFace Trainer
5. Evaluation (accuracy, F1, confusion matrix)
6. SHAP and LIME explainability
7. Multilingual prediction
8. Saving model to Google Drive / HuggingFace Hub

---

##  Known Limitations

- Model may misclassify sarcastic compliments as hate speech due to training data patterns — a known challenge in hate speech detection called **context-dependency**
- Hindi support relies on translation quality — very short or colloquial Hindi phrases may not translate accurately
- Dataset size (2K samples) is smaller than production-scale systems — performance would improve with more data

---

##  Links

-  **Live App**: [streamlit.app](https://hate-speech-detector-ewoeuogd9zxgehywempuje.streamlit.app/)
-  **Model**: [HuggingFace Hub](https://huggingface.co/alwinn/hate-speech-distilbert)
-  **Author**: [Alwin Paul](https://linkedin.com/in/alwin-paul-18a825249)
-  **Email**: alwinpaul694@gmail.com

---

##  License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

##  Acknowledgements

- [HuggingFace Transformers](https://huggingface.co/transformers/) for the DistilBERT model
- [SHAP](https://shap.readthedocs.io/) for explainability framework
- [Helsinki-NLP](https://huggingface.co/Helsinki-NLP) for translation models
- [Streamlit](https://streamlit.io/) for the web app framework
