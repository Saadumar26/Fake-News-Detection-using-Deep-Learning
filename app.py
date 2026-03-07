import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import time

# ─── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

# ─── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.stApp {
    background-color: #0a0a0f;
    color: #e8e8e8 !important;
}

p, span, div, label {
    color: #e8e8e8 !important;
}

.main-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3rem;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #ff6b35, #f7c59f);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.2rem;
}

.sub-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #aaaaaa !important;
    -webkit-text-fill-color: #aaaaaa !important;
    text-align: center;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}

.stTextArea label {
    color: #e8e8e8 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
}

.stTextArea textarea {
    background-color: #111118 !important;
    color: #e8e8e8 !important;
    border: 1px solid #2a2a3e !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.9rem !important;
    caret-color: #ff6b35 !important;
}

.stTextArea textarea::placeholder {
    color: #555 !important;
}

.stTextArea textarea:focus {
    border-color: #ff6b35 !important;
    box-shadow: 0 0 0 2px rgba(255,107,53,0.15) !important;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #ff6b35, #e85d27) !important;
    color: white !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 1px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.75rem 1rem !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(255,107,53,0.35) !important;
}

.result-fake {
    background: linear-gradient(135deg, #1a0a0a, #2d1010);
    border: 2px solid #ff3b3b;
    border-radius: 12px;
    padding: 28px;
    text-align: center;
    margin-top: 1.5rem;
}

.result-true {
    background: linear-gradient(135deg, #0a1a0a, #102d10);
    border: 2px solid #3bff7a;
    border-radius: 12px;
    padding: 28px;
    text-align: center;
    margin-top: 1.5rem;
}

.result-label-fake {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.5rem;
    color: #ff3b3b !important;
    -webkit-text-fill-color: #ff3b3b !important;
    letter-spacing: 4px;
}

.result-label-true {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.5rem;
    color: #3bff7a !important;
    -webkit-text-fill-color: #3bff7a !important;
    letter-spacing: 4px;
}

.confidence-text {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: #bbbbbb !important;
    -webkit-text-fill-color: #bbbbbb !important;
    margin-top: 8px;
}

.conf-bar-wrap {
    background: #1a1a2e;
    border-radius: 99px;
    height: 8px;
    margin: 12px 0;
    overflow: hidden;
}

.conf-bar-fake {
    height: 100%;
    background: linear-gradient(90deg, #ff3b3b, #ff7b7b);
    border-radius: 99px;
}

.conf-bar-true {
    height: 100%;
    background: linear-gradient(90deg, #3bff7a, #7bffa0);
    border-radius: 99px;
}

.stat-box {
    background: #111118;
    border: 1px solid #2a2a3e;
    border-radius: 8px;
    padding: 14px;
    text-align: center;
}

.stat-value {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 500;
    font-size: 1.2rem;
    color: #ff6b35 !important;
    -webkit-text-fill-color: #ff6b35 !important;
}

.stat-label {
    font-size: 0.7rem;
    color: #aaaaaa !important;
    -webkit-text-fill-color: #aaaaaa !important;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 4px;
}

hr {
    border: none;
    border-top: 1px solid #1e1e2e;
    margin: 2rem 0;
}

.warn-box {
    background: #1a1a0a;
    border: 1px solid #ff6b35;
    border-radius: 8px;
    padding: 14px 18px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #ff6b35 !important;
    -webkit-text-fill-color: #ff6b35 !important;
    margin-top: 1rem;
}

.footer-text {
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #888888 !important;
    -webkit-text-fill-color: #888888 !important;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─── Load Model ───────────────────────────────────────────────
HF_MODEL = "saadumar26/fake-news-detector"

@st.cache_resource
def load_model():
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(HF_MODEL)
        model = DistilBertForSequenceClassification.from_pretrained(HF_MODEL)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None


def predict(text, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256
    )
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    pred = torch.argmax(outputs.logits, dim=1).item()

    label = "TRUE" if pred == 1 else "FAKE"
    confidence = probs[pred].item() * 100
    fake_prob = probs[0].item() * 100
    true_prob = probs[1].item() * 100

    return label, confidence, fake_prob, true_prob


# ─── UI ───────────────────────────────────────────────────────
st.markdown('<div class="main-title">FAKE NEWS DETECTOR</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Powered by Deep Learning</div>', unsafe_allow_html=True)

# Load model
tokenizer, model = load_model()

# Input
news_text = st.text_area(
    "📰 Paste a news article or headline below:",
    height=200,
    placeholder="Enter news text here..."
)

word_count = len(news_text.split()) if news_text.strip() else 0

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-value">{word_count}</div>
        <div class="stat-label">Words</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-value">{len(news_text)}</div>
        <div class="stat-label">Characters</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("ANALYZE NEWS"):
    if not news_text.strip():
        st.warning("⚠️ Please enter some text before analyzing.")
    elif tokenizer is None or model is None:
        st.error("Model could not be loaded. Please check your connection and try again.")
    else:
        with st.spinner("Analyzing article..."):
            time.sleep(0.5)
            label, confidence, fake_prob, true_prob = predict(news_text, tokenizer, model)

        if label == "FAKE":
            st.markdown(f"""
            <div class="result-fake">
                <div style="font-size:2rem;">⚠️</div>
                <div class="result-label-fake">FAKE NEWS</div>
                <div class="confidence-text">Confidence: {confidence:.1f}%</div>
                <div class="conf-bar-wrap">
                    <div class="conf-bar-fake" style="width:{fake_prob:.1f}%"></div>
                </div>
                <div class="confidence-text">
                    FAKE: {fake_prob:.1f}% &nbsp;|&nbsp; TRUE: {true_prob:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-true">
                <div style="font-size:2rem;">✅</div>
                <div class="result-label-true">TRUE NEWS</div>
                <div class="confidence-text">Confidence: {confidence:.1f}%</div>
                <div class="conf-bar-wrap">
                    <div class="conf-bar-true" style="width:{true_prob:.1f}%"></div>
                </div>
                <div class="confidence-text">
                    TRUE: {true_prob:.1f}% &nbsp;|&nbsp; FAKE: {fake_prob:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="warn-box">
            ⚠️ DISCLAIMER: This is an AI-based tool. Always verify news through trusted sources before drawing conclusions.
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div class="footer-text">
    Fake News Detector &nbsp;·&nbsp; Final Year Project &nbsp;·&nbsp; Deep Learning
</div>
""", unsafe_allow_html=True)