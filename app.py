import streamlit as st
import pickle
import re
import nltk
import os

# ==============================
# NLTK
# ==============================
nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

# ==============================
# SAFE LOAD
# ==============================
def load_pickle(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_model():
    vectorizer = load_pickle('vectorizer.pkl')
    model = load_pickle('model.pkl')
    return vectorizer, model

vectorizer, model = load_model()

# ==============================
# PREPROCESS (🔥 MATCH TRAINING)
# ==============================
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def remove_sources(text):
    text = str(text).lower()
    for w in ['reuters', 'ap', 'cnn', 'bbc']:
        text = text.replace(w, '')
    return text

def preprocess(text):
    text = remove_sources(text)   # 🔥 important
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()
    words = [
        stemmer.stem(w)
        for w in words
        if w not in stop_words and len(w) > 2
    ]

    return ' '.join(words)

# ==============================
# PREDICT
# ==============================
def predict_news(text):
    clean = preprocess(text)

    if not clean.strip():
        return None, None

    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]

    # confidence
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(vec)[0]
        confidence = max(prob) * 100
    else:
        score = model.decision_function(vec)[0]
        confidence = min(abs(score) * 20 + 50, 99)

    return pred, confidence

# ==============================
# UI
# ==============================
st.title("📰 Fake News Detector")
st.markdown("### Achieved - 95% accuracy")
st.write("Paste new title and fews first lines to check")

text = st.text_area("News Text", height=200)

if st.button("Predict"):   # 🔥 changed

    if not text.strip():
        st.warning("Enter text")
    else:
        pred, conf = predict_news(text)

        if pred is None:
            st.error("Text too short")
        else:
            label = "REAL" if pred == 1 else "FAKE"

            if label == "REAL":
                st.success("✅ REAL NEWS")   # 🔥 no %
            else:
                st.error("❌ FAKE NEWS")     # 🔥 no %