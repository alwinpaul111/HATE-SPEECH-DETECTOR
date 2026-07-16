#  Context-Aware Hate Speech Detection using Transformer Models

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hate-speech-detector-ewoeuogd9zxgehywempuje.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/alwinn/hate-speech-distilbert)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A production-ready hate speech detection system built with DistilBERT, featuring rule-based sarcasm detection for indirect hate, SHAP explainability with confidence visualization, multilingual support (English + Hindi), and a deployed Streamlit web application.

---

##  Live Demo

** [Try the app here](https://hate-speech-detector-ewoeuogd9zxgehywempuje.streamlit.app/)**

** Model on HuggingFace: [alwinn/hate-speech-distilbert](https://huggingface.co/alwinn/hate-speech-distilbert)**

---

## Screenshots

| Input | Prediction | Confidence |
|---|---|---|
| "All immigrants should go back" |  Hate Speech | 99.9% |
| "I love how diverse our community is" | No Hate | 99.3% |
| "Oh sure, they're GREAT... if you like crime" |  Indirect Hate (Sarcasm) | 85% sarcasm confidence |
| "तुम बहुत घटिया इंसान हो" |  Hate Speech | 99.5% (auto-translated) |

---

##  Features

| Feature | Description |
|---|---|
|  **DistilBERT Fine-tuning** | Transformer model fine-tuned on a balanced 2K hate speech dataset |
|  **Sarcasm Detection** | Two-layer rule-based engine: keyword matching + regex patterns for indirect hate |
|  **SHAP Explainability** | Bar chart showing top words pushing toward/away from hate speech |
|  **Confidence Pie Chart** | Visual breakdown of hate vs safe probability |
|  **Prediction Summary Table** | Structured view of prediction, confidence, language, translation, context |
|  **Multilingual** | English + Hindi support via Helsinki-NLP translation pipeline |
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
       ├──────────────────────────────────┐
       ▼                                  ▼
DistilBERT Classifier            Rule-Based Sarcasm Engine
       │                         (keyword + regex patterns)
       ▼                                  │
┌──────┴──────┐                           │
│             │                           ▼
▼             ▼                   Sarcastic? ──YES──► ⚠️ Indirect Hate
Hate       No Hate ◄──────────────────────┘
Speech
       │
       ▼
SHAP Bar Chart + Confidence Pie
       │
       ▼
Streamlit Web App Output
```

---

##  Sarcasm Detection — How It Works

Most hate speech detectors miss **indirect hate** — hostility disguised as neutral or positive language. This system uses a **two-layer rule-based engine** that is more reliable than ML-only approaches for common sarcasm patterns:

**Layer 1 — Keyword Matching** (92% confidence threshold)
Catches phrases like: `"oh sure"`, `"what could go wrong"`, `"those people are always"`, `"if you like crime"`, `"so trustworthy, right?"`

**Layer 2 — Regex Pattern Matching** (85% confidence threshold)
Catches structural patterns:
- Ellipsis + contradiction: `"they're GREAT... if you like crime"`
- ALL-CAPS emphasis + negative framing: `"GREAT for the neighborhood... crime"`
- Sarcastic question endings: `"always so trustworthy, right?"`

**Output:** Explains *why* sarcasm was detected, e.g.:
> *"Matched sarcastic phrase: 'if you like crime'"*
> *"Caps emphasis (GREAT) with negative framing"*

---

##  Tech Stack

| Category | Tools |
|---|---|
| **Model** | DistilBERT (HuggingFace Transformers) |
| **Fine-tuning** | HuggingFace Trainer API |
| **Explainability** | SHAP (bar chart visualization) |
| **Sarcasm Detection** | Rule-based: keyword matching + regex patterns |
| **Translation** | Helsinki-NLP/opus-mt-hi-en |
| **Visualization** | Matplotlib (bar chart + pie chart) |
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

##  Getting Started

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

## Model Details

| Parameter | Value |
|---|---|
| Base Model | `distilbert-base-uncased` |
| Dataset | hate_speech_dataset_2000 (balanced, 2,025 samples) |
| Train/Val/Test Split | 70% / 15% / 15% |
| Training Epochs | 3 |
| Batch Size | 16 |
| Max Sequence Length | 128 tokens |
| Optimizer | AdamW (weight_decay=0.01) |
| Labels | 0 = No Hate, 1 = Hate Speech |
| Pipeline | `top_k=None` for calibrated probability outputs |

### Dataset Label Distribution
```
No Hate      : 1,010 samples (50%)
Hate Speech  : 1,015 samples (50%)
```
Perfectly balanced — no class imbalance issues.

---

##  Results

| Metric | Score |
|---|---|
| Accuracy | ~92% |
| F1 Score (weighted) | ~0.91 |
| Hate Speech Precision | ~0.93 |
| Hate Speech Recall | ~0.91 |

---

##  SHAP Explainability

SHAP (SHapley Additive exPlanations) shows **which words** drove the prediction and by how much:

- 🔴 **Red bars** → word increases hate speech probability
- 🔵 **Blue bars** → word decreases hate speech probability
- Bar length = strength of influence

Example for *"All immigrants should go back to their country"*:
- `immigrants`, `go back`, `country` → longest red bars
- Model correctly identifies xenophobic language patterns

---

##  Multilingual Support

Hindi input is automatically translated then classified:

```
Input (Hindi) → langdetect → Helsinki-NLP MT → DistilBERT → Prediction
```

Example:
```
Input      : "तुम बहुत घटिया इंसान हो!"
Translated : "You are a very mean person!"
Result     : 🚨 Hate Speech (99.5% confidence)
```

---

##  How to Train Your Own Model

Open the notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alwinpaul111/HATE-SPEECH-DETECTOR/blob/main/notebooks/hate_speech_detection.ipynb)

The notebook covers:
1. Dataset loading and label mapping
2. Text preprocessing
3. DistilBERT tokenization
4. Fine-tuning with HuggingFace Trainer
5. Evaluation (accuracy, F1, confusion matrix)
6. SHAP explainability
7. Multilingual prediction pipeline
8. Saving model to Google Drive / HuggingFace Hub

---

##  Known Limitations

- **Sarcasm rule-engine** covers common patterns but cannot detect all forms of implicit hate — novel sarcasm structures may be missed
- **Hindi translation quality** depends on Helsinki-NLP — very short or highly colloquial Hindi may not translate accurately
- **Dataset size** (2K samples) is smaller than production-scale systems — a larger dataset would improve generalization
- **Model calibration** — confidence is appropriately lower (~65-75%) on genuinely ambiguous inputs, which is expected behavior

---

##  Links

- 🌐 **Live App**: [streamlit.app](https://hate-speech-detector-ewoeuogd9zxgehywempuje.streamlit.app/)
- 🤗 **Model**: [HuggingFace Hub](https://huggingface.co/alwinn/hate-speech-distilbert)
- 👤 **Author**: [Alwin Paul](https://linkedin.com/in/alwin-paul-18a825249)
- 📧 **Email**: alwinpaul694@gmail.com

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [HuggingFace Transformers](https://huggingface.co/transformers/) for the DistilBERT model and Trainer API
- [SHAP](https://shap.readthedocs.io/) for the explainability framework
- [Helsinki-NLP](https://huggingface.co/Helsinki-NLP) for the Hindi→English translation model
- [Streamlit](https://streamlit.io/) for the web app framework
