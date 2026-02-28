import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st


# ---------------- MODEL + DATA ----------------
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

model = load_model('sentiment_rnn_imdb.h5')


# ---------------- FUNCTIONS ----------------
def preprocess_text(text):
    words = text.lower().split()
    word_indices = [word_index.get(w, 2) + 3 for w in words]
    return sequence.pad_sequences([word_indices], maxlen=500)


def predict_sentiment(text):
    processed_text = preprocess_text(text)
    prediction = model.predict(processed_text)

    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    score = prediction[0][0]

    return sentiment, score


# ---------------- STREAMLIT UI ----------------
st.title("IMDB Movie Review Sentiment Analysis")

user_input = st.text_area(
    "Enter a movie review:",
    key="review_input"
)

if st.button("Predict Sentiment"):

    if user_input.strip() == "":
        st.warning("Please enter a movie review.")
    else:
        sentiment, score = predict_sentiment(user_input)

        st.write(f"Sentiment: {sentiment}")
        st.write(f"Confidence Score: {score:.4f}")