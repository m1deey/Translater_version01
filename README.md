# 🌐 Translator Version 01

A neural machine translation web app: type a short English sentence, get an instant Spanish translation. Built from scratch using an LSTM Encoder-Decoder architecture with an Attention mechanism.

## How it works
1. Type a short English sentence
2. Hit Translate
3. Get the Spanish translation instantly

Note: works best on short sentences (under 15 words) — the model was trained on short sentence pairs.

## Tech
- Model: LSTM Encoder-Decoder with Attention mechanism
- Framework: TensorFlow / Keras
- Frontend: Streamlit
- Model hosting: Hugging Face Hub
- Deployment: Streamlit Community Cloud

## Try it live
https://3ckska.streamlit.app/

## Run locally
pip install -r requirements.txt
streamlit run app.py

## Model details
- Trained on ~105,000 English-Spanish sentence pairs
- Vocabulary: English ~22k words, Spanish ~40k words
- Max sequence length: 15 tokens
- Architecture: Embedding + LSTM encoder, LSTM decoder, Attention layer
- ~68% validation accuracy

---
Built by [Yasser](https://github.com/m1deey)
