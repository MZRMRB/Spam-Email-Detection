# %%
# *Spam Email Detection	*

# Goal: Build a model to classify emails as spam or non-spam.
# Steps:

# Frontend: Streamlit.
# Dataset: SMS Spam Collection Dataset
# Model: Use Logistic Regression, Naive Bayes, or LSTM for text classification.
# Evaluation: Precision, recall, and F1-score.

# https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset 

# Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional
from tensorflow.keras.optimizers import Adam

# Load dataset
df = pd.read_csv('spam.csv', usecols=[0, 1], names=['label', 'message'], header=0, encoding='latin1')

# Preprocessing: Convert labels to binary values (0 for ham, 1 for spam)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Machine Learning Models: Using TF-IDF Vectorization
# Create TF-IDF Vectorizer to convert text data into numerical form
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)

# Fit and transform the training data, transform the testing data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Standardize for logistic regression
scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train_tfidf)
X_test_scaled = scaler.transform(X_test_tfidf)

# Train ML models
models = {
    'Naïve Bayes': MultinomialNB(),
    'SVM': SVC(kernel='linear', C=1.0),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Logistic Regression': LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
}
 
# Train and evaluate each machine learning model
results = {}
for name, model in models.items():
    if name == 'Naive Bayes':  # Naive Bayes works directly with sparse matrices
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
    else:  # Other models require dense arrays
        model.fit(X_train_tfidf.toarray(), y_train)
        y_pred = model.predict(X_test_tfidf.toarray())

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    results[name] = {'accuracy': accuracy, 'report': report}

# Deep Learning Model: Using LSTM for Sequence Classification
# Tokenize messages and pad sequences for deep learning models
max_features = 5000  # Maximum number of words to consider
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(df['message'].values)

# Convert text to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train.values)
X_test_seq = tokenizer.texts_to_sequences(X_test.values)

# Pad sequences to ensure uniform input size
max_len = 20  # Maximum length of each sequence
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# Build an LSTM-based neural network model
model = Sequential([
        Embedding(max_features, 256, input_length=max_len),
        SpatialDropout1D(0.3),
        Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)),
        Dense(1, activation='sigmoid')
    ]) # Output layer for binary classification

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train the model
model.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_data=(X_test_pad, y_test), verbose=2)

# Evaluate the LSTM model
y_pred_dl = (model.predict(X_test_pad) > 0.5).astype("int32")
dl_accuracy = accuracy_score(y_test, y_pred_dl)
dl_report = classification_report(y_test, y_pred_dl, zero_division=0)
results['Deep Learning (LSTM)'] = {'accuracy': dl_accuracy, 'report': dl_report}

# Print results for all models
print("\n=== SPAM DETECTION MODEL RESULTS ===\n")
for model_name, result in results.items():
    print(f"Model: {model_name}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print("Classification Report:\n", result['report'])
    print("=" * 50)