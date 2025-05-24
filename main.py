import numpy as np
import tensorflow as tf
import tensorflow.keras .preprocessing as sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN, Dense
from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import load_model
import os

model_path = 'SimpleRNN_imdb.h5'
if os.path.exists(model_path):
	model = load_model(model_path)
else:
	print(f"Model file '{model_path}' not found. Please check the file path.")
	
max_features = 10000
word_index= imdb.get_word_index()
reverse_word_index={values: key for (key, values) in word_index.items()}

def decode_review(text):
    return' '.join([reverse_word_index.get(i - 3, '?') for i in text])

def preprocess_text(text):
    if isinstance(text, np.ndarray):
        # If already a (1, 500) shaped padded input, return it
        if text.shape == (1, 500):
            return text
        elif text.size == 1:
            text = text.item()
        else:
            raise ValueError(f"Expected a string or (1, 500) array, got shape {text.shape} with contents: {text}")

    if not isinstance(text, str):
        raise TypeError(f"Expected input to be a string, got {type(text)}")

    words = text.lower().split()
    encoded_text = [word_index.get(word, 2) + 3 for word in words]
    padded_text = pad_sequences([encoded_text], maxlen=500)
    return padded_text



def predict_sentiment(text):
	preprocessed_text = preprocess_text(text)
	prediction = model.predict(preprocessed_text)
	sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
	return sentiment, prediction[0][0]

import streamlit as st
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (positive or negative):")
user_input = st.text_area("Movie Review", "Type your review here...")
if st.button("Predict Sentiment"):
    if user_input:
        preprocessed_text = preprocess_text(user_input)
        sentiment, score = predict_sentiment(preprocessed_text)
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Confidence Score: {score:.2f}")
    else:
        st.write("Please enter a movie review.")

