import streamlit as st
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import SimpleRNN

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# -------------------------------------------------
# SESSION STATE MANAGEMENT (For Clear Button)
# -------------------------------------------------
if 'user_input_text' not in st.session_state:
    st.session_state['user_input_text'] = ''

def clear_text():
    st.session_state['user_input_text'] = ''

def set_example_positive():
    st.session_state['user_input_text'] = "This movie was absolutely fantastic! The acting was great and the plot kept me on the edge of my seat."

def set_example_negative():
    st.session_state['user_input_text'] = "I hated this film. It was a complete waste of time and the script was terrible."

# -------------------------------------------------
# CUSTOM CLASS (Fix for TF Version Mismatches)
# -------------------------------------------------
# Keep this if you are moving between different TF versions (e.g. Colab -> Local)
class SimpleRNNSafe(SimpleRNN):
    def __init__(self, *args, **kwargs):
        kwargs.pop("time_major", None)
        super().__init__(*args, **kwargs)

# -------------------------------------------------
# LOAD MODEL & WORD INDEX
# -------------------------------------------------
@st.cache_resource
def load_sentiment_model():
    # Note: Loading the specific file name you saved in the notebook
    model = load_model(
        "sudha_simple_rnn_model.h5", 
        custom_objects={"SimpleRNN": SimpleRNNSafe},
        compile=False # We don't need to compile for inference, speeds up loading
    )
    word_index = imdb.get_word_index()
    return model, word_index

try:
    model, word_index = load_sentiment_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------
def preprocess_text(text, maxlen=500):
    words = text.lower().split()
    # Using the +3 offset to match IMDB dataset indexing (0,1,2 reserved)
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=maxlen
    )
    return padded_review

def predict_sentiment(text):
    processed = preprocess_text(text)
    prediction = model.predict(processed, verbose=0)
    score = prediction[0][0]
    
    sentiment = "Positive" if score >= 0.5 else "Negative"
    
    # Calculate confidence (distance from 0.5 decision boundary)
    if sentiment == "Positive":
        confidence = score
    else:
        confidence = 1 - score
        
    return sentiment, score, confidence

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.header("⚙️ Model Info")
    st.write("Architecture: **Simple RNN**")
    st.write("Dataset: **IMDB Reviews**")
    st.markdown("---")
    st.write("This model was trained to classify movie reviews as either positive or negative based on sentiment analysis.")

# -------------------------------------------------
# MAIN UI
# -------------------------------------------------
st.title("🎬 Movie Review Sentiment")
st.subheader("Simple RNN Analysis")

# Quick Test Buttons
st.write("Don't want to type? Try an example:")
col_ex1, col_ex2 = st.columns(2)
with col_ex1:
    st.button("😊 Load Positive Review", on_click=set_example_positive)
with col_ex2:
    st.button("😡 Load Negative Review", on_click=set_example_negative)

# Input Area
user_input = st.text_area(
    "Enter your review here:",
    height=150,
    placeholder="Type something...",
    key='user_input_text' # Binds this widget to session state
)

col1, col2 = st.columns([1, 1])
with col1:
    analyze = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
with col2:
    # On click, this runs the clear_text function to reset session state
    st.button("🧹 Clear Text", on_click=clear_text, use_container_width=True)

# -------------------------------------------------
# PREDICTION LOGIC
# -------------------------------------------------
if analyze:
    if not user_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("🔎 Analyzing sentiment..."):
            # Small delay so spinner is visible (UX best practice)
            time.sleep(0.3)

            sentiment, score, confidence = predict_sentiment(user_input)

            # Ensure native Python floats (Streamlit-safe)
            score = float(score)
            confidence = float(confidence)

        st.markdown("---")

        # Result Display
        if sentiment == "Positive":
            st.success(f"✅ **Sentiment: Positive**")
        else:
            st.error(f"❌ **Sentiment: Negative**")

        # Metrics
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Confidence", f"{confidence * 100:.2f}%")
        with m_col2:
            st.metric("Model Score", f"{score:.4f}")

        # Polarity Bar
        st.caption("Sentiment Polarity (0 = Negative, 1 = Positive)")
        st.progress(score)

st.markdown("---")
st.caption("Built with TensorFlow & Streamlit")