import streamlit as st
import pickle
import numpy as np
from spam_email_detector import models, tfidf_vectorizer, tokenizer, model, pad_sequences, max_len

def predict_spam_ml(email, model_choice):
    """Predicts whether an email is spam or not using traditional ML models."""
    email_tfidf = tfidf_vectorizer.transform([email])
    if model_choice == "Naïve Bayes":
        prediction = models["Naïve Bayes"].predict(email_tfidf)[0]
    else:
        prediction = models[model_choice].predict(email_tfidf.toarray())[0]
    return prediction

def predict_spam_dl(email):
    """Predicts whether an email is spam or not using the LSTM model."""
    email_seq = tokenizer.texts_to_sequences([email])
    email_pad = pad_sequences(email_seq, maxlen=max_len)
    prediction = (model.predict(email_pad) > 0.5).astype("int32")[0][0]
    return prediction

# Streamlit UI
st.title("📩 Spam Email Detection")
st.write("Enter an email message below and check if it's spam or not.")

# User Input
user_input = st.text_area("Paste your email here:")

# Model Selection
model_choice = st.selectbox("Choose Model:", ["Naïve Bayes", "SVM", "Random Forest", "Logistic Regression", "LSTM (Deep Learning)"])

# Predict Button
if st.button("Check for Spam"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text before predicting!")
    else:
        if model_choice == "LSTM (Deep Learning)":
            prediction = predict_spam_dl(user_input)
        else:
            prediction = predict_spam_ml(user_input, model_choice)

        if prediction == 1:
            st.error("🚨 **Spam Email Detected!** 🚨")
        else:
            st.success("✅ **Not Spam!** Safe to Read.")
