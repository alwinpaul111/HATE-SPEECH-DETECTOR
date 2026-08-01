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
    page_icon=" ",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .summary-table {
        width: 100%; border-collapse: collapse;
        margin: 1rem 0; font-size: 0.95rem;
        border-radius: 8px; overflow: hidden;
    }
    .summary-table th {
        background-color: #1e2030; color: #a0aec0;
        font-weight: 600; text-transform: uppercase;
        font-size: 0.75rem; letter-spacing: 0.08em;
        padding: 10px 16px; text-align: left;
        border-bottom: 1px solid #2d3748;
    }
    .summary-table td {
        padding: 10px 16px;
        border-bottom: 1px solid #2d3748; color: #e2e8f0;
    }
    .summary-table tr:last-child td { border-bottom: none; }
    .summary-table tr:hover td { background-color: #1a1f35; }
    .badge-hate {
        background: #e53e3e22; color: #fc8181;
        border: 1px solid #e53e3e55; padding: 2px 10px;
        border-radius: 999px; font-weight: 600; font-size: 0.85rem;
    }
    .badge-safe {
        background: #38a16922; color: #68d391;
        border: 1px solid #38a16955; padding: 2px 10px;
        border-radius: 999px; font-weight: 600; font-size: 0.85rem;
    }
    .badge-sarcasm-hate {
        background: #e53e3e22; color: #fc8181;
        border: 1px solid #e53e3e55; padding: 2px 10px;
        border-radius: 999px; font-weight: 600; font-size: 0.85rem;
    }
    .badge-sarcasm-general {
        background: #d69e2e22; color: #f6e05e;
        border: 1px solid #d69e2e55; padding: 2px 10px;
        border-radius: 999px; font-weight: 600; font-size: 0.85rem;
    }
    .badge-direct {
        background: #38a16922; color: #68d391;
        border: 1px solid #38a16955; padding: 2px 10px;
        border-radius: 999px; font-weight: 600; font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH = "alwinn/hate-speech-distilbert"

POSITIVE_PHRASES = [
     #  positive phrases
    "lovely person", "wonderful person", "great person",
    "amazing person", "beautiful day", "love you",
    "you are great", "you are kind", "well done",
    "good job", "you are amazing", "you are wonderful",
    "you are lovely", "such a good", "such a great",
    "you are such a nice", "you are such a good",
    "i love this", "i love how", "i love our",
    "excellent work", "proud of you",
    "you are the best", "appreciate you", "thank you", "grateful",
    "good person", "nice person", "is a good", "is a nice",
    "they are good", "they are nice", "is a great", "is a wonderful",

    #  Anti-hate statements (condemning violence/hate, not promoting it)
    "violence against any group", "violence is never acceptable",
    "violence against any", "never acceptable",
    "is never acceptable", "is not acceptable",
    "should never be allowed to discriminate",
    "discrimination is wrong", "racism is wrong",
    "hate is wrong", "hatred is wrong",
    "every person deserves", "everyone deserves",
    "deserves dignity", "deserves respect",
    "dignity and respect", "equal rights",
    "regardless of race", "regardless of religion",
    "regardless of their", "regardless of background",
    "no place for hate", "no place for racism",
    "stand against hate", "fight against hate",
    "condemn this behavior", "this is unacceptable",
    "we must do better", "we should do better",
    "hate has no place", "love not hate",

    #  "I hate X" where X is a harmful concept, not a group
    "i hate terrorism", "i hate violence",
    "i hate injustice", "i hate racism",
    "i hate discrimination", "i hate corruption",
    "i hate poverty", "i hate war",
    "hate terrorists", "hate terrorism",
    "hate criminals", "hate crime",
    "hate bullying", "hate abuse",
]

# ── Sarcasm Patterns ──────────────────────────────────────────────────────────
# TYPE 1: HATEFUL SARCASM — positive framing hiding group hostility
HATEFUL_SARCASM_KEYWORDS = [
    "oh sure", "yeah right", "wow what a surprise",
    "oh of course", "totally trustworthy", "so trustworthy",
    "oh absolutely", "what could go wrong", "another one of them",
    "those people are always", "them causing", "if you like crime",
    "love all the noise", "great addition", "yeah because those",
    "keep letting them",
]
HATEFUL_SARCASM_PATTERNS = [
    r'\b[A-Z]{2,}\b.*\b(if you like|right\?|sure|yeah|wow|totally)',
    r'^(oh sure|yeah right|wow what|oh wow|oh absolutely|great idea|totally|yeah because|oh of course)',
    r'(right\?|isn\'t it\?|don\'t they\?|do they\?|what could go wrong)',
    r'\.\.\.\s*(if you like|not|unless|except)',
    r'\b(those|them|these)\s+people\b.*(always|never|so|such|really)',
    r'another one of\s+(them|those|these)',
    r'what a (surprise|shock|shocker)',
    r'(keep letting|just let|why do we let|letting them).*(in|come|stay)',
]

# TYPE 2: GENERAL SARCASM — mocking individual incompetence (non-hateful)
GENERAL_SARCASM_KEYWORDS = [
    "wow you're a genius", "truly revolutionary",
    "congratulations you've", "congratulations, you've",
    "what a brilliant idea", "if your goal was failure",
    "new level of incompetence", "consistently wrong",
    "zero effort maximum", "zero effort, maximum",
    "talent for missing", "completely misplaced",
    "you really thought", "was a smart move",
    "making everyone's day harder", "you deserve an award",
    "you managed to make", "simplest task difficult",
    "confidence is inspiring", "misplaced confidence",
    "you're a genius", "truly impressive",
    "award for being",
]
GENERAL_SARCASM_PATTERNS = [
    r'what a (brilliant|great|wonderful|fantastic|amazing|genius).*(if|but|though|however|was)',
    r'(truly|really|absolutely|definitely)\s+(revolutionary|impressive|amazing|brilliant|genius)',
    r'congratulations.*(incompetence|failure|wrong|bad|terrible|awful)',
    r'you deserve.*(award|prize|medal).*(wrong|bad|incompetence|failure|missing)',
    r'^wow[,\s].*(you|your).*(genius|brilliant|smart|impressive|amazing)',
    r'zero\s+\w+[,\s]+maximum\s+\w+',
    r'your\s+(confidence|talent|ability|skill).*(misplaced|unmatched|inspiring)',
    r'you\s+managed\s+to\s+make.*(difficult|hard|worse|fail)',
    r'you\s+really\s+thought\s+that\s+was',
]


def rule_based_sarcasm(text: str) -> tuple[bool, str, float, str]:
    """
    Returns (is_sarcastic, sarcasm_type, confidence, reason)
    sarcasm_type: "hateful" | "general" | "none"
    """
    lower = text.lower()

    # ── Check hateful sarcasm first ───────────────────────────────────────────
    for kw in HATEFUL_SARCASM_KEYWORDS:
        if kw in lower:
            return True, "hateful", 0.92, f'Matched hateful sarcasm phrase: "{kw}"'
    for pattern in HATEFUL_SARCASM_PATTERNS:
        if re.search(pattern, lower):
            return True, "hateful", 0.85, "Matched hateful sarcasm pattern"

    # ALL-CAPS + negative group framing
    caps_words = [w for w in re.findall(r'\b[A-Z]{3,}\b', text)
                  if w not in ['URL', 'USER']]
    if caps_words:
        group_negative = ['crime', 'criminal', 'problem', 'issue', 'trouble',
                          'dangerous', 'terrible', 'awful', 'horrible']
        if any(nw in lower for nw in group_negative):
            return True, "hateful", 0.78, \
                f'Caps emphasis ({", ".join(caps_words)}) with negative group framing'

    # Ellipsis + contradiction with group reference
    if '...' in text:
        parts = text.split('...')
        if len(parts) >= 2 and len(parts[1].strip()) > 0:
            group_words = ['they', 'them', 'those', 'people',
                           'immigrants', 'refugees', 'outsiders']
            if any(gw in lower for gw in group_words):
                return True, "hateful", 0.72, "Ellipsis contradiction with group reference"

    # ── Check general sarcasm ─────────────────────────────────────────────────
    for kw in GENERAL_SARCASM_KEYWORDS:
        if kw in lower:
            return True, "general", 0.90, f'Matched sarcasm phrase: "{kw}"'
    for pattern in GENERAL_SARCASM_PATTERNS:
        if re.search(pattern, lower):
            return True, "general", 0.82, "Matched general sarcasm pattern"

    # Backhanded compliment: positive + negative in same sentence
    positive_w = ['brilliant', 'genius', 'impressive', 'amazing',
                  'revolutionary', 'inspiring', 'unmatched']
    negative_w = ['incompetence', 'failure', 'wrong', 'misplaced',
                  'difficult', 'harder', 'missing', 'consistently']
    if any(pw in lower for pw in positive_w) and any(nw in lower for nw in negative_w):
        return True, "general", 0.75, \
            "Backhanded compliment: positive + negative framing in same sentence"

    return False, "none", 0.0, "No sarcasm signals found"


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
        inputs = hi_tok([text], return_tensors="pt",
                        padding=True, truncation=True, max_length=128)
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
        MODEL_PATH, config=config)
    model.config.id2label = {0: "No Hate", 1: "Hate Speech"}
    model.config.label2id = {"No Hate": 0, "Hate Speech": 1}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    clf = pipeline("text-classification", model=model,
                   tokenizer=tokenizer, device=-1, top_k=None)
    return clf, tokenizer

@st.cache_resource(show_spinner=False)
def load_translation_model():
    from transformers import MarianMTModel, MarianTokenizer
    hi_tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-hi-en")
    hi_mod = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-hi-en")
    return hi_tok, hi_mod

# ── Prediction ────────────────────────────────────────────────────────────────
def get_probs(clf, text: str) -> tuple[str, float, float]:
    results = clf(text[:512])[0]
    scores  = {r["label"]: r["score"] for r in results}
    hate    = scores.get("Hate Speech", 0.0)
    safe    = scores.get("No Hate", 1.0 - hate)
    label   = "Hate Speech" if hate > 0.5 else "No Hate"
    return label, hate, safe

# ── SHAP Bar Chart ────────────────────────────────────────────────────────────
def render_shap_barchart(clf, text: str):
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
    vals    = shap_values[0, :, 1].values
    words   = shap_values[0, :, 1].data
    top_n   = min(10, len(vals))
    indices = np.argsort(np.abs(vals))[-top_n:]
    top_words = [words[i] for i in indices]
    top_vals  = [vals[i]  for i in indices]
    colors    = ["#e53e3e" if v > 0 else "#4299e1" for v in top_vals]

    fig, ax = plt.subplots(figsize=(9, max(3, top_n * 0.5)))
    fig.patch.set_facecolor("#1a1f35")
    ax.set_facecolor("#1a1f35")
    bars = ax.barh(top_words, top_vals, color=colors, edgecolor="none", height=0.6)
    for bar, val in zip(bars, top_vals):
        ax.text(
            bar.get_width() + (0.002 if val >= 0 else -0.002),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}", va="center",
            ha="left" if val >= 0 else "right",
            color="#e2e8f0", fontsize=9
        )
    ax.axvline(0, color="#4a5568", linewidth=1.2)
    ax.set_xlabel("SHAP Value", color="#a0aec0", fontsize=9)
    ax.set_title("Top Words by Impact on Prediction", color="#e2e8f0",
                 fontsize=11, pad=10)
    ax.tick_params(colors="#e2e8f0", labelsize=10)
    ax.spines[["top","right","bottom","left"]].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(color="#e53e3e", label="→ Increases Hate probability"),
            Patch(color="#4299e1", label="→ Decreases Hate probability"),
        ],
        loc="lower right", framealpha=0.2, labelcolor="#e2e8f0", fontsize=8
    )
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── Confidence Pie ────────────────────────────────────────────────────────────
def render_confidence_pie(hate_prob: float, safe_prob: float):
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    fig.patch.set_facecolor("#1a1f35")
    ax.set_facecolor("#1a1f35")
    wedges, _ = ax.pie(
        [hate_prob, safe_prob], colors=["#e53e3e", "#38a169"],
        startangle=90, wedgeprops={"edgecolor": "#1a1f35", "linewidth": 2}
    )
    ax.legend(wedges, [f"Hate\n{hate_prob:.1%}", f"Safe\n{safe_prob:.1%}"],
              loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2,
              labelcolor="#e2e8f0", framealpha=0, fontsize=9)
    ax.set_title("Confidence", color="#e2e8f0", fontsize=10, pad=8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── UI ────────────────────────────────────────────────────────────────────────
st.title(" Context-Aware Hate Speech Detector")
st.markdown("**Powered by DistilBERT + SHAP + Sarcasm Detection + Multilingual (EN + HI)**")
st.markdown("---")

with st.spinner("Loading models... (first load ~30 sec)"):
    clf, tokenizer = load_hate_model()
    hi_tok, hi_mod = load_translation_model()

col1, col2 = st.columns([2, 1])
with col1:
    user_input = st.text_area(" Enter text (English or Hindi):",
                               height=150, placeholder="Type any text here...")
with col2:
    st.markdown("###  Options")
    show_shap  = st.checkbox("Show SHAP Bar Chart", value=True)
    show_pie   = st.checkbox("Show Confidence Pie Chart", value=True)
    show_trans = st.checkbox("Show Translation (Hindi)", value=True)

analyze = st.button(" Analyze", type="primary", use_container_width=True)

# ── Analysis ──────────────────────────────────────────────────────────────────
if analyze and user_input.strip():
    with st.spinner("Analyzing..."):

        # Language detection + translation
        lang            = detect_lang(user_input)
        translated      = None
        text_to_predict = user_input

        if lang == "hi":
            translated      = translate_hi_to_en(user_input, hi_tok, hi_mod)
            text_to_predict = translated

        cleaned = clean_text(text_to_predict)

        # ✅ Run sarcasm detection FIRST before model
        is_sarcastic, sarc_type, sarc_conf, sarc_reason = rule_based_sarcasm(user_input)

        # ✅ Decision logic — order matters
        is_obvious_positive = any(p in cleaned.lower() for p in POSITIVE_PHRASES)

        if is_obvious_positive:
            # Known positive phrases → always No Hate
            label, hate_prob, safe_prob = "No Hate", 0.03, 0.97

        elif is_sarcastic and sarc_type == "general":
            # General sarcasm (mocking incompetence) → always No Hate
            # Don't call the model — these are NEVER hate speech by definition
            label, hate_prob, safe_prob = "No Hate", 0.05, 0.95

        else:
            # Run the model for everything else
            label, hate_prob, safe_prob = get_probs(clf, cleaned)

        # Final context decision
        if is_sarcastic and sarc_type == "hateful" and label == "No Hate" and sarc_conf >= 0.72:
            final_context = "indirect_hate"
        elif is_sarcastic and sarc_type == "general":
            final_context = "general_sarcasm"
        elif is_sarcastic and sarc_type == "hateful":
            final_context = "hateful_sarcasm_low"
        else:
            final_context = "direct"

    # ── Banner ────────────────────────────────────────────────────────────────
    st.markdown("---")
    if label == "Hate Speech":
        st.error(f"🚨 HATE SPEECH DETECTED — {hate_prob:.1%} confidence")
    elif final_context == "indirect_hate":
        st.warning(
            f"⚠️ POSSIBLE INDIRECT HATE SPEECH — Sarcasm detected ({sarc_conf:.0%} confidence)  \n"
            f"Base model predicted **No Hate**, but sarcastic framing suggests hidden hostility."
        )
    elif final_context == "general_sarcasm":
        st.info(
            f"💡 SARCASM DETECTED — Mocking/ironic tone, but not hate speech ({sarc_conf:.0%} confidence)"
        )
    else:
        st.success(f"✅ NO HATE SPEECH DETECTED — {safe_prob:.1%} confidence")

    st.progress(float(hate_prob))

    # ── Table + Pie ───────────────────────────────────────────────────────────
    left, right = st.columns([3, 1])
    with left:
        st.markdown("#### 📋 Prediction Summary")

        badge = (
            '<span class="badge-hate">Hate Speech</span>'
            if label == "Hate Speech"
            else '<span class="badge-safe">No Hate</span>'
        )
        if final_context == "indirect_hate":
            ctx_badge = '<span class="badge-sarcasm-hate">⚠️ Hateful Sarcasm</span>'
        elif final_context == "general_sarcasm":
            ctx_badge = '<span class="badge-sarcasm-general">💡 General Sarcasm</span>'
        else:
            ctx_badge = '<span class="badge-direct">✅ Direct</span>'

        lang_display  = {"en": "🇬🇧 English", "hi": "🇮🇳 Hindi"}.get(lang, lang.upper())
        trans_display = f"<em>{translated}</em>" if translated else "—"
        conf_val      = f"{hate_prob:.1%}" if label == "Hate Speech" else f"{safe_prob:.1%}"
        sarc_display  = f"{sarc_conf:.0%} ({sarc_type})" if is_sarcastic else "—"

        st.markdown(f"""
        <table class="summary-table">
            <thead><tr><th>Feature</th><th>Value</th></tr></thead>
            <tbody>
                <tr><td>Prediction</td><td>{badge}</td></tr>
                <tr><td>Confidence</td><td>{conf_val}</td></tr>
                <tr><td>Hate Probability</td><td>{hate_prob:.1%}</td></tr>
                <tr><td>Safe Probability</td><td>{safe_prob:.1%}</td></tr>
                <tr><td>Language</td><td>{lang_display}</td></tr>
                <tr><td>Translation</td><td>{trans_display}</td></tr>
                <tr><td>Context</td><td>{ctx_badge}</td></tr>
                <tr><td>Sarcasm Confidence</td><td>{sarc_display}</td></tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)

    with right:
        if show_pie:
            render_confidence_pie(hate_prob, safe_prob)

    # ── Context Awareness ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🎭 Context Awareness")

    if final_context == "indirect_hate":
        st.warning(
            f"⚠️ **Hateful Sarcasm detected** ({sarc_conf:.0%} confidence)  \n"
            f"**Reason:** {sarc_reason}  \n\n"
            f"This text uses sarcastic or ironic language to express hostility toward "
            f"a group while appearing neutral on the surface. The base model predicted "
            f"**No Hate** because the literal words are not explicitly hateful — but the "
            f"sarcastic framing indicates **indirect hate speech**."
        )
    elif final_context == "general_sarcasm":
        st.info(
            f"💡 **General Sarcasm detected** ({sarc_conf:.0%} confidence)  \n"
            f"**Reason:** {sarc_reason}  \n\n"
            f"This text uses sarcasm or irony to mock someone's competence or actions, "
            f"but is **not directed at any protected group**. It is not classified as "
            f"hate speech — sarcasm alone does not make content hateful."
        )
    elif final_context == "hateful_sarcasm_low":
        st.info(
            f"💡 **Weak sarcasm signal** ({sarc_conf:.0%} confidence)  \n"
            f"**Reason:** {sarc_reason}  \n\n"
            f"Some sarcasm indicators present but below threshold. "
            f"Base prediction **{label}** is likely correct."
        )
    else:
        st.info(
            f"✅ **No sarcasm detected** — text appears direct and literal.  \n"
            f"Base prediction **{label}** is reliable."
        )

    # ── SHAP ──────────────────────────────────────────────────────────────────
    if show_shap and not is_obvious_positive and final_context != "general_sarcasm":
        st.markdown("---")
        st.markdown("#### 🔍 SHAP Feature Importance — Top Words")
        st.caption(
            "🔴 **Red bars** = word increases Hate Speech probability  "
            "| 🔵 **Blue bars** = word decreases Hate Speech probability"
        )
        with st.spinner("Generating SHAP analysis (~30 sec)..."):
            try:
                render_shap_barchart(clf, cleaned)
            except Exception as e:
                st.warning(f"SHAP analysis failed: {e}")
    elif show_shap and (is_obvious_positive or final_context == "general_sarcasm"):
        st.info("💡 SHAP skipped — prediction made by context rules, not the model.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Built with HuggingFace Transformers · SHAP · Streamlit  |  "
    "Model: [alwinn/hate-speech-distilbert](https://huggingface.co/alwinn/hate-speech-distilbert)"
)
