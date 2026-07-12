import streamlit as st
import numpy as np
import re
import os
import json
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import (
    pipeline, AutoTokenizer,
    AutoModelForSequenceClassification, AutoConfig
)
from langdetect import detect, LangDetectException

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hate Speech Detector",
    page_icon="🔍",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Summary table styling */
    .summary-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        font-size: 0.95rem;
        border-radius: 8px;
        overflow: hidden;
    }
    .summary-table th {
        background-color: #1e2030;
        color: #a0aec0;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.08em;
        padding: 10px 16px;
        text-align: left;
        border-bottom: 1px solid #2d3748;
    }
    .summary-table td {
        padding: 10px 16px;
        border-bottom: 1px solid #2d3748;
        color: #e2e8f0;
    }
    .summary-table tr:last-child td {
        border-bottom: none;
    }
    .summary-table tr:hover td {
        background-color: #1a1f35;
    }
    .badge-hate {
        background: #e53e3e22;
        color: #fc8181;
        border: 1px solid #e53e3e55;
        padding: 2px 10px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-safe {
        background: #38a16922;
        color: #68d391;
        border: 1px solid #38a16955;
        padding: 2px 10px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .shap-section {
        background: #1a1f35;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
    }
    .section-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #718096;
        margin-bottom: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH = "alwinn/hate-speech-distilbert"

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
    "appreciate you", "thank you", "grateful",
    "good person", "nice person", "he is good", "she is good",
    "he is nice", "she is nice", "is a good", "is a nice",
    "they are good", "they are nice", "is a great", "is a wonderful",
    "is a lovely", "is a kind", "is a fantastic", "is an amazing"
]

# ── Helper Functions ───────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "[URL]", text)
    text = re.sub(r"@\w+", "[USER]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def detect_lang(text: str) -> str:
    try:
        return detect(str(text))
    except LangDetectException:
        return "en"

def translate_hi_to_en(text: str, hi_tok, hi_mod) -> str:
    try:
        inputs = hi_tok(
            [text], return_tensors="pt",
            padding=True, truncation=True, max_length=128
        )
        translated = hi_mod.generate(**inputs)
        return hi_tok.decode(translated[0], skip_special_tokens=True)
    except Exception:
        return text

# ── Model Loaders ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_hate_model():
    config = AutoConfig.from_pretrained(MODEL_PATH)
    config.id2label = {0: "No Hate", 1: "Hate Speech"}
    config.label2id = {"No Hate": 0, "Hate Speech": 1}
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, config=config
    )
    model.config.id2label = {0: "No Hate", 1: "Hate Speech"}
    model.config.label2id = {"No Hate": 0, "Hate Speech": 1}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # top_k=None returns ALL class scores → correct probability outputs
    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1,
        top_k=None
    )
    return clf, tokenizer

@st.cache_resource(show_spinner=False)
def load_translation_model():
    from transformers import MarianMTModel, MarianTokenizer
    hi_tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-hi-en")
    hi_mod = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-hi-en")
    return hi_tok, hi_mod

# ── Prediction Logic ──────────────────────────────────────────────────────────
def get_probabilities(clf, text: str) -> dict:
    """
    Returns {label: probability} dict using top_k=None output.
    top_k=None gives ALL classes, so we always have both scores.
    """
    results = clf(text[:512])[0]   # list of {label, score} dicts
    return {r["label"]: r["score"] for r in results}

def predict(clf, text: str) -> tuple[str, float, float]:
    """Returns (label, hate_prob, safe_prob)"""
    probs = get_probabilities(clf, text)
    hate_prob = probs.get("Hate Speech", 0.0)
    safe_prob = probs.get("No Hate", 1.0 - hate_prob)
    label = "Hate Speech" if hate_prob > 0.5 else "No Hate"
    return label, hate_prob, safe_prob

def predict_fn_for_shap(clf, texts):
    """
    predict_fn for SHAP — returns np.array of shape (n, 2).
    Column 0 = P(No Hate), Column 1 = P(Hate Speech).
    Uses top_k=None so probabilities are always correct.
    """
    results = clf(list(texts), truncation=True, max_length=128)
    probs = []
    for result in results:
        scores = {r["label"]: r["score"] for r in result}
        hate = scores.get("Hate Speech", 0.0)
        safe = scores.get("No Hate", 1.0 - hate)
        probs.append([safe, hate])
    return np.array(probs)

# ── SHAP Visualization ────────────────────────────────────────────────────────
def render_shap(clf, text: str):
    """Renders SHAP word-level explanation as HTML (fixes blank graph bug)."""
    fn = lambda texts: predict_fn_for_shap(clf, texts)
    explainer = shap.Explainer(fn, shap.maskers.Text(r"\W+"))
    shap_values = explainer([text], max_evals=200)

    # Use matplotlib figure — more reliable in Streamlit than shap HTML
    fig, ax = plt.subplots(figsize=(13, 2.2))
    ax.axis("off")
    plt.tight_layout(pad=0)

    # Get word-level SHAP for class 1 (Hate Speech)
    vals  = shap_values[0, :, 1].values
    words = shap_values[0, :, 1].data

    # Render as colored text using matplotlib
    x = 0.01
    y = 0.5
    max_abs = max(abs(vals).max(), 1e-6)
    for word, val in zip(words, vals):
        intensity = min(abs(val) / max_abs, 1.0)
        if val > 0:
            color = (1, 1 - intensity * 0.85, 1 - intensity * 0.85)  # red
        else:
            color = (1 - intensity * 0.85, 1 - intensity * 0.65, 1)  # blue
        txt = ax.text(
            x, y, word + " ",
            ha="left", va="center",
            fontsize=13, fontweight="bold" if abs(val) > 0.3 * max_abs else "normal",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=color, alpha=0.85, linewidth=0),
            transform=ax.transAxes
        )
        # Advance x position
        renderer = fig.canvas.get_renderer()
        bb = txt.get_window_extent(renderer=renderer)
        x += (bb.width / fig.get_size_inches()[0] / fig.dpi) + 0.008
        if x > 0.95:
            x = 0.01
            y -= 0.35

    fig.patch.set_facecolor("#1a1f35")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🔍 Context-Aware Hate Speech Detector")
st.markdown("**Powered by DistilBERT + SHAP Explainability + Multilingual (EN + HI)**")
st.markdown("---")

# Load models
with st.spinner("Loading models... (first load ~30 sec)"):
    clf, tokenizer = load_hate_model()
    hi_tok, hi_mod = load_translation_model()

# Input layout
col1, col2 = st.columns([2, 1])
with col1:
    user_input = st.text_area(
        "✍️ Enter text (English or Hindi):",
        height=150,
        placeholder="Type any text here..."
    )
with col2:
    st.markdown("### ⚙️ Options")
    show_shap  = st.checkbox("Show SHAP Explanation", value=True)
    show_trans = st.checkbox("Show Translation (Hindi)", value=True)

analyze = st.button("🔍 Analyze", type="primary", use_container_width=True)

# ── Analysis ──────────────────────────────────────────────────────────────────
if analyze and user_input.strip():
    with st.spinner("Analyzing..."):

        # Language detection
        lang       = detect_lang(user_input)
        translated = None
        text_to_predict = user_input

        # Translate if Hindi
        if lang == "hi":
            translated      = translate_hi_to_en(user_input, hi_tok, hi_mod)
            text_to_predict = translated

        cleaned = clean_text(text_to_predict)

        # Positive phrase override
        is_obvious_positive = any(
            phrase in cleaned.lower() for phrase in POSITIVE_PHRASES
        )

        if is_obvious_positive:
            label     = "No Hate"
            hate_prob = 0.03
            safe_prob = 0.97
        else:
            label, hate_prob, safe_prob = predict(clf, cleaned)

    # ── Result Banner ─────────────────────────────────────────────────────────
    st.markdown("---")
    if label == "Hate Speech":
        st.error(f"🚨 HATE SPEECH DETECTED — {hate_prob:.1%} confidence")
    else:
        st.success(f"✅ NO HATE SPEECH DETECTED — {safe_prob:.1%} confidence")

    st.progress(float(hate_prob))

    # ── Professional Summary Table ────────────────────────────────────────────
    st.markdown("#### 📋 Prediction Summary")

    badge = (
        '<span class="badge-hate">Hate Speech</span>'
        if label == "Hate Speech"
        else '<span class="badge-safe">No Hate</span>'
    )

    lang_display = {
        "en": "🇬🇧 English", "hi": "🇮🇳 Hindi"
    }.get(lang, lang.upper())

    trans_display = f"<em>{translated}</em>" if translated else "—"
    conf_display  = f"{hate_prob:.1%}" if label == "Hate Speech" else f"{safe_prob:.1%}"

    table_html = f"""
    <table class="summary-table">
        <thead>
            <tr>
                <th>Feature</th>
                <th>Value</th>
            </tr>
        </thead>
        <tbody>
            <tr><td>Prediction</td><td>{badge}</td></tr>
            <tr><td>Confidence</td><td>{conf_display}</td></tr>
            <tr><td>Hate Probability</td><td>{hate_prob:.1%}</td></tr>
            <tr><td>Safe Probability</td><td>{safe_prob:.1%}</td></tr>
            <tr><td>Language</td><td>{lang_display}</td></tr>
            <tr><td>Translation</td><td>{trans_display}</td></tr>
        </tbody>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    # ── SHAP Explanation ──────────────────────────────────────────────────────
    if show_shap and not is_obvious_positive:
        st.markdown("---")
        st.markdown("#### 🔍 SHAP Word-Level Explanation")
        st.caption(
            "🔴 **Red** = word pushes prediction toward **Hate Speech**  "
            "| 🔵 **Blue** = word pushes toward **No Hate**"
        )
        with st.spinner("Generating SHAP explanation (~30 sec)..."):
            try:
                render_shap(clf, cleaned)
            except Exception as e:
                st.warning(f"SHAP visualization failed: {e}")
    elif show_shap and is_obvious_positive:
        st.info(
            "💡 SHAP skipped — text matched a known positive phrase pattern."
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Built with HuggingFace Transformers · SHAP · Streamlit  |  "
    "Model: [alwinn/hate-speech-distilbert](https://huggingface.co/alwinn/hate-speech-distilbert)"
)
