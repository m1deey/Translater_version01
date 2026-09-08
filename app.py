import streamlit as st
import pickle
import re
import urllib.request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 15
MODEL_URL = "https://huggingface.co/M1deey/translator_/resolve/main/translator_en_es.h5"

st.set_page_config(page_title="Translator Version 01", page_icon="🌐", layout="centered")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(160deg, #0f0f1a 0%, #1a1a2e 100%);
    color: #e8e8e8;
}
h1 {
    text-align: center;
    font-family: 'Georgia', serif;
    font-weight: 700;
    background: linear-gradient(90deg, #d4af37, #f4e5c2, #d4af37);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 1px;
}
.subtitle {
    text-align: center;
    color: #a0a0b0;
    font-family: 'Georgia', serif;
    font-size: 0.95em;
    margin-bottom: 2em;
}
.stTextInput > div > div > input {
    background-color: #1e1e30;
    color: white;
    border: 1px solid #d4af37;
    border-radius: 8px;
    padding: 12px;
    font-size: 16px;
}
div.stButton > button {
    background: linear-gradient(90deg, #d4af37, #b8942e);
    color: #0f0f1a;
    border-radius: 8px;
    border: none;
    padding: 12px 30px;
    font-weight: 700;
    letter-spacing: 0.5px;
    width: 100%;
    transition: 0.3s;
}
div.stButton > button:hover {
    background: linear-gradient(90deg, #f4e5c2, #d4af37);
    transform: scale(1.01);
}
.result-box {
    background: #1e1e30;
    border-left: 4px solid #d4af37;
    padding: 20px;
    border-radius: 8px;
    margin-top: 20px;
    font-size: 1.3em;
    font-family: 'Georgia', serif;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    urllib.request.urlretrieve(MODEL_URL, "translator_en_es.h5")
    model = load_model("translator_en_es.h5")
    with open("eng_tokenizer.pkl", "rb") as f:
        eng_tokenizer = pickle.load(f)
    with open("spa_tokenizer.pkl", "rb") as f:
        spa_tokenizer = pickle.load(f)
    return model, eng_tokenizer, spa_tokenizer

with st.spinner("Loading translation engine..."):
    model, eng_tokenizer, spa_tokenizer = load_assets()

def translate(sentence):
    sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence).strip().lower()
    seq = eng_tokenizer.texts_to_sequences([sentence])
    seq = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    target_seq = np.zeros((1, MAX_LEN))
    target_seq[0, 0] = spa_tokenizer.word_index["<start>"]

    result = []
    for i in range(1, MAX_LEN):
        preds = model.predict([seq, target_seq], verbose=0)
        next_id = np.argmax(preds[0, i-1])
        word = spa_tokenizer.index_word.get(next_id, "")
        if word == "<end>" or word == "":
            break
        result.append(word)
        target_seq[0, i] = next_id

    return " ".join(result)

st.markdown("<h1>Translator Version 01</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>English → Spanish · Neural Machine Translation</p>", unsafe_allow_html=True)

text = st.text_input("", placeholder="Type a short English sentence...")

if st.button("Translate"):
    if text.strip():
        with st.spinner("Translating..."):
            result = translate(text)
        st.markdown(f"<div class='result-box'>{result}</div>", unsafe_allow_html=True)
    else:
        st.warning("Please enter a sentence.")

st.markdown("<br><p style='text-align:center; color:#555; font-size:0.8em;'>Built with LSTM + Attention · Trained on short sentence pairs</p>", unsafe_allow_html=True)
