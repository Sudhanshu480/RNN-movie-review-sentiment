import streamlit as st
import numpy as np
import tensorflow as tf
import time

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model 
from tensorflow.keras.layers import SimpleRNN 

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# =====================================================
# SESSION STATE
# =====================================================
if "user_input_text" not in st.session_state:
    st.session_state.user_input_text = ""

def clear_text():
    st.session_state.user_input_text = ""

def set_example_positive():
    st.session_state.user_input_text = (
        "This movie was absolutely fantastic. "
        "The acting was brilliant and the story was engaging."
    )

def set_example_negative():
    st.session_state.user_input_text = (
        "I hated this movie. It was boring, poorly written, "
        "and a complete waste of time."
    )

# =====================================================
# SAFE SimpleRNN (TF VERSION FIX)
# =====================================================
from tensorflow.keras.layers import InputLayer

class InputLayerSafe(InputLayer):
    def __init__(self, *args, **kwargs):
        kwargs.pop("batch_shape", None)
        super().__init__(*args, **kwargs)

class SimpleRNNSafe(SimpleRNN):
    def __init__(self, *args, **kwargs):
        kwargs.pop("time_major", None)
        super().__init__(*args, **kwargs)

# =====================================================
# CONSTANTS (MUST MATCH TRAINING)
# =====================================================
NUM_WORDS = 10000
MAX_LEN = 500
OOV_INDEX = 2
INDEX_OFFSET = 3

# =====================================================
# LOAD MODEL & WORD INDEX
# =====================================================
@st.cache_resource
def load_artifacts():
    model = load_model(
        "sudha_simple_rnn_model.h5",
        custom_objects={
        "SimpleRNN": SimpleRNNSafe,
        "InputLayer": InputLayerSafe
        },
        compile=False
    )
    word_index = imdb.get_word_index()
    return model, word_index

try:
    model, word_index = load_artifacts()
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# =====================================================
# TEXT PREPROCESSING (FIXED & SAFE)
# =====================================================
def preprocess_text(text: str):
    words = text.lower().split()
    encoded = []

    for word in words:
        idx = word_index.get(word, OOV_INDEX)
        if idx >= NUM_WORDS:
            idx = OOV_INDEX
        encoded.append(idx + INDEX_OFFSET)

    padded = sequence.pad_sequences(
        [encoded],
        maxlen=MAX_LEN
    )
    return padded

# =====================================================
# PREDICTION
# =====================================================
def predict_sentiment(text: str):
    processed = preprocess_text(text)
    score = float(model.predict(processed, verbose=0)[0][0])

    if score >= 0.5:
        sentiment = "Positive"
        confidence = score
    else:
        sentiment = "Negative"
        confidence = 1 - score

    return sentiment, score, confidence

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.header("⚙️ Model Info")
    st.write("Architecture: **Simple RNN**")
    st.write("Dataset: **IMDB Reviews**")
    st.write("Vocabulary Size: **10,000**")
    st.write("Sequence Length: **500**")
    st.markdown("---")
    st.write("This app performs sentiment analysis on movie reviews.")

# =====================================================
# MAIN UI
# =====================================================
st.title("🎬 Movie Review Sentiment Analysis")
st.subheader("Simple RNN | End-to-End NLP App")

st.write("Try an example or enter your own review:")

ex1, ex2 = st.columns(2)
with ex1:
    st.button("😊 Positive Example", on_click=set_example_positive)
with ex2:
    st.button("😡 Negative Example", on_click=set_example_negative)

user_input = st.text_area(
    "Enter your review:",
    height=160,
    key="user_input_text",
    placeholder="Type a movie review here..."
)

col1, col2 = st.columns(2)
with col1:
    analyze = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
with col2:
    st.button("🧹 Clear Text", on_click=clear_text, use_container_width=True)

# =====================================================
# OUTPUT
# =====================================================
if analyze:
    if not user_input.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        with st.spinner("Analyzing sentiment..."):
            time.sleep(0.3)  # UX delay
            sentiment, score, confidence = predict_sentiment(user_input)

        st.markdown("---")

        if sentiment == "Positive":
            st.success("✅ **Positive Review**")
        else:
            st.error("❌ **Negative Review**")

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Confidence", f"{confidence * 100:.2f}%")
        with m2:
            st.metric("Model Score", f"{score:.4f}")

        st.caption("Sentiment Polarity (0 = Negative, 1 = Positive)")
        st.progress(score)

st.markdown("---")
st.caption("Built with TensorFlow & Streamlit | Simple RNN IMDB Sentiment App")
