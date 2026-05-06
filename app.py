
import streamlit as st
import numpy as np
import re
import os
import json
import torch
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from transformers import (
    pipeline, AutoTokenizer,
    AutoModelForSequenceClassification, AutoConfig
)
from langdetect import detect

st.set_page_config(page_title="Hate Speech Detector", page_icon="🔍", layout="wide")
st.title("🔍 Context-Aware Hate Speech Detector")
st.markdown("**Powered by DistilBERT + SHAP Explainability + Multilingual (EN + HI)**")
st.markdown("---")

MODEL_PATH = "alwinn/hate-speech-distilbert"

@st.cache_resource

def load_model():
    config          = AutoConfig.from_pretrained(MODEL_PATH)
    config.id2label = {0: "No Hate", 1: "Hate Speech"}
    config.label2id = {"No Hate": 0, "Hate Speech": 1}
    model           = AutoModelForSequenceClassification.from_pretrained(
                          MODEL_PATH, config=config)
    model.config.id2label = {0: "No Hate", 1: "Hate Speech"}
    tokenizer       = AutoTokenizer.from_pretrained(MODEL_PATH)
    clf             = pipeline(
                          "text-classification",
                          model=model,
                          tokenizer=tokenizer,
                          device=-1
                      )
    return clf, tokenizer

@st.cache_resource
def load_translation_model():
    from transformers import MarianMTModel, MarianTokenizer
    hi_tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-hi-en")
    hi_mod = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-hi-en")
    return hi_tok, hi_mod

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "[URL]", text)
    text = re.sub(r"@\w+", "[USER]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def detect_lang(text):
    try:
        return detect(str(text))
    except:
        return "en"

def translate_hi_to_en(text, hi_tok, hi_mod):
    inputs     = hi_tok([text], return_tensors="pt", padding=True, truncation=True)
    translated = hi_mod.generate(**inputs)
    return hi_tok.decode(translated[0], skip_special_tokens=True)

with st.spinner("Loading model... (first load ~30 sec)"):
    clf, tokenizer = load_model()
    hi_tok, hi_mod = load_translation_model()

col1, col2 = st.columns([2, 1])
with col1:
    user_input = st.text_area("✍️ Enter text (English or Hindi):", height=150,
                               placeholder="Type any text here...")
with col2:
    st.markdown("### ⚙️ Options")
    show_shap  = st.checkbox("Show SHAP Explanation", value=True)
    show_trans = st.checkbox("Show Translation (Hindi)", value=True)

analyze = st.button("🔍 Analyze", type="primary", use_container_width=True)

if analyze and user_input.strip():
    with st.spinner("Analyzing..."):
        lang            = detect_lang(user_input)
        translated      = None
        text_to_predict = user_input

        if lang == "hi":
            translated      = translate_hi_to_en(user_input, hi_tok, hi_mod)
            text_to_predict = translated
cleaned = clean_text(text_to_predict)

        # ── Quick fix for obvious positive phrases ──────────
POSITIVE_PHRASES = [
            "lovely person", "wonderful person", "great person",
            "amazing person", "beautiful day", "love you",
            "you are great", "you are kind", "well done",
            "good job", "you are amazing", "you are wonderful",
            "you are lovely", "such a good", "such a great",
            "you are such a nice", "you are such a good",
            "i love this", "i love how", "i love our",
            "beautiful", "fantastic", "excellent work",
            "proud of you", "you are the best",
            "appreciate you", "thank you", "grateful"
        ]

is_obvious_positive = any(
            phrase in cleaned.lower() for phrase in POSITIVE_PHRASES
        )

if is_obvious_positive:
            label      = "No Hate"
            score      = 0.97
            hate_score = 0.03
        else:
            result     = clf(cleaned[:512])[0]
            label      = result["label"]
            score      = result["score"]
            hate_score = score if label == "Hate Speech" else 1 - score
        # ── End of quick fix ────────────────────────────────

        st.markdown("---")
        if label == "Hate Speech":
            st.error(f"🚨 HATE SPEECH DETECTED — {hate_score:.1%} confidence")
        else:
            st.success(f"✅ NO HATE SPEECH — {(1-hate_score):.1%} confidence")

        m1, m2, m3 = st.columns(3)
        m1.metric("Prediction", label)
        m2.metric("Hate Score", f"{hate_score:.1%}")
        m3.metric("Language",   lang.upper())
        st.progress(float(hate_score))

        if lang == "hi" and translated and show_trans:
            st.info(f"🔄 Translated: **{translated}**")

        if show_shap:
            st.markdown("### 🔍 SHAP Explanation")
            st.caption("🔴 Red = Hate | 🔵 Blue = Safe")

            def predict_fn(texts):
                out   = clf(list(texts), truncation=True, max_length=128)
                probs = []
                for r in out:
                    s = r["score"]
                    probs.append([1-s, s] if r["label"] == "Hate Speech" else [s, 1-s])
                return np.array(probs)

            with st.spinner("Generating SHAP (~30 sec)..."):
                explainer = shap.Explainer(predict_fn, shap.maskers.Text(r"\W+"))
                shap_vals = explainer([cleaned], max_evals=200)
                fig, ax   = plt.subplots(figsize=(12, 2))
                shap.plots.text(shap_vals[0, :, 1], display=False)
                st.pyplot(fig)
                plt.close()

st.markdown("---")
st.caption("Built with HuggingFace Transformers + SHAP + Streamlit")
