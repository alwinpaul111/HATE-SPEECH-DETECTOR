import streamlit as st
import numpy as np
import re
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
    .summary-table tr:last-child td { border-bottom: none; }
    .summary-table tr:hover td { background-color: #1a1f35; }
    .badge-hate {
        background: #e53e3e22; color: #fc8181;
        border: 1px solid #e53e3e55;
        padding: 2px 10px; border-radius: 999px;
        font-weight: 600; font-size: 0.85rem;
    }
    .badge-safe {
        background: #38a16922; color: #68d391;
        border: 1px solid #38a16955;
        padding: 2px 10px; border-radius: 999px;
        font-weight: 600; font-size: 0.85rem;
    }
    .badge-sarcasm {
        background: #d69e2e22; color: #f6e05e;
        border: 1px solid #d69e2e55;
        padding: 2px 10px; border-radius: 999px;
        font-weight: 600; font-size: 0.85rem;
    }
    .context-box {
        background: #1a1f35;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
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
    "fantastic", "excellent work", "proud of you",
    "you are the best", "appreciate you", "thank you", "grateful",
    "good person", "nice person", "he is good", "she is good",
    "he is nice", "she is nice", "is a good", "is a nice",
    "they are good", "they are nice", "is a great", "is a wonderful",
    "is a lovely", "is a kind", "is a fantastic", "is an amazing"
]

# ── Helpers ───────────────────────────────────────────────────────────────────
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
        out = hi_mod.generate(**inputs)
        return hi_tok.decode(out[0], skip_special_tokens=True)
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
    clf = pipeline(
        "text-classification", model=model,
        tokenizer=tokenizer, device=-1, top_k=None
    )
    return clf, tokenizer

@st.cache_resource(show_spinner=False)
def load_translation_model():
    from transformers import MarianMTModel, MarianTokenizer
    hi_tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-hi-en")
    hi_mod = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-hi-en")
    return hi_tok, hi_mod

@st.cache_resource(show_spinner=False)
def load_sarcasm_model():
    try:
        sarc = pipeline(
            "text-classification",
            model="mrm8488/t5-base-finetuned-sarcasm-twitter",
            device=-1
        )
        return sarc
    except Exception:
        return None

# ── Prediction ────────────────────────────────────────────────────────────────
def get_probs(clf, text: str) -> tuple[str, float, float]:
    results = clf(text[:512])[0]
    scores  = {r["label"]: r["score"] for r in results}
    hate    = scores.get("Hate Speech", 0.0)
    safe    = scores.get("No Hate", 1.0 - hate)
    label   = "Hate Speech" if hate > 0.5 else "No Hate"
    return label, hate, safe

def check_sarcasm(sarc_model, text: str) -> tuple[bool, float]:
    if sarc_model is None:
        return False, 0.0
    try:
        result       = sarc_model(text[:512])[0]
        is_sarcastic = result["label"].lower() == "sarcasm"
        return is_sarcastic, round(result["score"], 3)
    except Exception:
        return False, 0.0

# ── SHAP Bar Chart ────────────────────────────────────────────────────────────
def render_shap_barchart(clf, text: str):
    """
    Renders SHAP as a horizontal bar chart — far more reliable than
    word-level highlighting. Shows top words pushing toward/away from hate.
    """
    def predict_fn(texts):
        out   = clf(list(texts), truncation=True, max_length=128)
        probs = []
        for result in out:
            scores = {r["label"]: r["score"] for r in result}
            hate   = scores.get("Hate Speech", 0.0)
            safe   = scores.get("No Hate", 1.0 - hate)
            probs.append([safe, hate])
        return np.array(probs)

    explainer   = shap.Explainer(predict_fn, shap.maskers.Text(r"\W+"))
    shap_values = explainer([text], max_evals=200)

    # Extract word-level SHAP values for class 1 (Hate Speech)
    vals  = shap_values[0, :, 1].values
    words = shap_values[0, :, 1].data

    # Sort by absolute importance, take top 10
    top_n   = min(10, len(vals))
    indices = np.argsort(np.abs(vals))[-top_n:]
    top_words = [words[i] for i in indices]
    top_vals  = [vals[i]  for i in indices]

    # Colors: red for positive SHAP (→ hate), blue for negative (→ safe)
    colors = ["#e53e3e" if v > 0 else "#4299e1" for v in top_vals]

    # Plot horizontal bar chart
    fig, ax = plt.subplots(figsize=(9, max(3, top_n * 0.5)))
    fig.patch.set_facecolor("#1a1f35")
    ax.set_facecolor("#1a1f35")

    bars = ax.barh(top_words, top_vals, color=colors, edgecolor="none", height=0.6)

    # Labels on bars
    for bar, val in zip(bars, top_vals):
        x_pos = bar.get_width() + (0.002 if val >= 0 else -0.002)
        ha    = "left" if val >= 0 else "right"
        ax.text(
            x_pos, bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}", va="center", ha=ha,
            color="#e2e8f0", fontsize=9
        )

    ax.axvline(0, color="#4a5568", linewidth=1.2)
    ax.set_xlabel("SHAP Value (impact on Hate Speech probability)",
                  color="#a0aec0", fontsize=9)
    ax.set_title("Top Words by Impact", color="#e2e8f0", fontsize=11, pad=10)
    ax.tick_params(colors="#e2e8f0", labelsize=10)
    ax.spines[["top","right","bottom","left"]].set_visible(False)
    ax.xaxis.label.set_color("#a0aec0")

    # Legend
    from matplotlib.patches import Patch
    legend = ax.legend(
        handles=[
            Patch(color="#e53e3e", label="→ Pushes toward Hate Speech"),
            Patch(color="#4299e1", label="→ Pushes toward No Hate"),
        ],
        loc="lower right", framealpha=0.2,
        labelcolor="#e2e8f0", fontsize=8
    )

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── Confidence Pie Chart ──────────────────────────────────────────────────────
def render_confidence_pie(hate_prob: float, safe_prob: float):
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    fig.patch.set_facecolor("#1a1f35")
    ax.set_facecolor("#1a1f35")

    sizes  = [hate_prob, safe_prob]
    colors = ["#e53e3e", "#38a169"]
    labels = [f"Hate\n{hate_prob:.1%}", f"Safe\n{safe_prob:.1%}"]

    wedges, texts = ax.pie(
        sizes, colors=colors, startangle=90,
        wedgeprops={"edgecolor": "#1a1f35", "linewidth": 2}
    )
    ax.legend(
        wedges, labels, loc="lower center",
        bbox_to_anchor=(0.5, -0.08), ncol=2,
        labelcolor="#e2e8f0", framealpha=0, fontsize=9
    )
    ax.set_title("Confidence", color="#e2e8f0", fontsize=10, pad=8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🔍 Context-Aware Hate Speech Detector")
st.markdown("**Powered by DistilBERT + SHAP Explainability + Sarcasm Detection + Multilingual (EN + HI)**")
st.markdown("---")

with st.spinner("Loading models... (first load ~30 sec)"):
    clf, tokenizer = load_hate_model()
    hi_tok, hi_mod = load_translation_model()
    sarc_model     = load_sarcasm_model()

col1, col2 = st.columns([2, 1])
with col1:
    user_input = st.text_area(
        "✍️ Enter text (English or Hindi):",
        height=150, placeholder="Type any text here..."
    )
with col2:
    st.markdown("### ⚙️ Options")
    show_shap  = st.checkbox("Show SHAP Bar Chart", value=True)
    show_pie   = st.checkbox("Show Confidence Pie Chart", value=True)
    show_trans = st.checkbox("Show Translation (Hindi)", value=True)

analyze = st.button("🔍 Analyze", type="primary", use_container_width=True)

# ── Analysis ──────────────────────────────────────────────────────────────────
if analyze and user_input.strip():
    with st.spinner("Analyzing..."):
        lang            = detect_lang(user_input)
        translated      = None
        text_to_predict = user_input

        if lang == "hi":
            translated      = translate_hi_to_en(user_input, hi_tok, hi_mod)
            text_to_predict = translated

        cleaned = clean_text(text_to_predict)

        # Positive phrase override
        is_obvious_positive = any(
            phrase in cleaned.lower() for phrase in POSITIVE_PHRASES
        )

        if is_obvious_positive:
            label, hate_prob, safe_prob = "No Hate", 0.03, 0.97
        else:
            label, hate_prob, safe_prob = get_probs(clf, cleaned)

        # Context: sarcasm detection
        is_sarcastic, sarc_conf = check_sarcasm(sarc_model, cleaned)
        context_label = "No Hate"
        if is_sarcastic and label == "No Hate" and sarc_conf > 0.7:
            context_label = "⚠️ Possible Indirect Hate (Sarcasm Detected)"
        else:
            context_label = label

    # ── Banner ────────────────────────────────────────────────────────────────
    st.markdown("---")
    if label == "Hate Speech":
        st.error(f"🚨 HATE SPEECH DETECTED — {hate_prob:.1%} confidence")
    elif "Indirect Hate" in context_label:
        st.warning(f"⚠️ POSSIBLE INDIRECT HATE (SARCASM DETECTED) — {sarc_conf:.1%} sarcasm confidence")
    else:
        st.success(f"✅ NO HATE SPEECH DETECTED — {safe_prob:.1%} confidence")

    st.progress(float(hate_prob))

    # ── Two column layout: table + pie ────────────────────────────────────────
    left, right = st.columns([3, 1])

    with left:
        st.markdown("#### 📋 Prediction Summary")

        badge = (
            '<span class="badge-hate">Hate Speech</span>'
            if label == "Hate Speech"
            else '<span class="badge-safe">No Hate</span>'
        )
        context_badge = (
            '<span class="badge-sarcasm">⚠️ Sarcasm Detected</span>'
            if is_sarcastic
            else '<span class="badge-safe">Direct</span>'
        )
        lang_display  = {"en": "🇬🇧 English", "hi": "🇮🇳 Hindi"}.get(lang, lang.upper())
        trans_display = f"<em>{translated}</em>" if translated else "—"
        conf_display  = f"{hate_prob:.1%}" if label == "Hate Speech" else f"{safe_prob:.1%}"

        st.markdown(f"""
        <table class="summary-table">
            <thead><tr><th>Feature</th><th>Value</th></tr></thead>
            <tbody>
                <tr><td>Prediction</td><td>{badge}</td></tr>
                <tr><td>Confidence</td><td>{conf_display}</td></tr>
                <tr><td>Hate Probability</td><td>{hate_prob:.1%}</td></tr>
                <tr><td>Safe Probability</td><td>{safe_prob:.1%}</td></tr>
                <tr><td>Language</td><td>{lang_display}</td></tr>
                <tr><td>Translation</td><td>{trans_display}</td></tr>
                <tr><td>Context</td><td>{context_badge}</td></tr>
                <tr><td>Sarcasm Confidence</td><td>{"—" if not is_sarcastic else f"{sarc_conf:.1%}"}</td></tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)

    with right:
        if show_pie:
            render_confidence_pie(hate_prob, safe_prob)

    # ── Context Awareness Box ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🎭 Context Awareness")
    if is_sarcastic and sarc_conf > 0.7:
        st.warning(
            f"⚠️ **Sarcasm detected** ({sarc_conf:.1%} confidence) — "
            f"this text may contain **indirect or implicit hate speech** "
            f"disguised as a compliment or neutral statement. "
            f"The base model predicted **{label}**, but context signals suggest caution."
        )
    elif is_sarcastic and sarc_conf <= 0.7:
        st.info(
            f"💡 Low-confidence sarcasm signal detected ({sarc_conf:.1%}) — "
            f"text appears mostly direct. Base prediction: **{label}**."
        )
    else:
        st.info(
            f"✅ **No sarcasm detected** — text appears direct and literal. "
            f"Base prediction **{label}** is reliable."
        )

    # ── SHAP Bar Chart ────────────────────────────────────────────────────────
    if show_shap and not is_obvious_positive:
        st.markdown("---")
        st.markdown("#### 🔍 SHAP Feature Importance — Top Words")
        st.caption(
            "🔴 **Red bars** = word increases hate speech probability  "
            "| 🔵 **Blue bars** = word decreases hate speech probability  "
            "| Bar length = strength of influence"
        )
        with st.spinner("Generating SHAP analysis (~30 sec)..."):
            try:
                render_shap_barchart(clf, cleaned)
            except Exception as e:
                st.warning(f"SHAP analysis failed: {e}")
    elif show_shap and is_obvious_positive:
        st.info("💡 SHAP skipped — text matched a known positive phrase.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Built with HuggingFace Transformers · SHAP · Streamlit  |  "
    "Model: [alwinn/hate-speech-distilbert](https://huggingface.co/alwinn/hate-speech-distilbert)"
)
