# data_prep.py

import pandas as pd
import numpy as np
import re
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from torch.utils.data import TensorDataset, DataLoader

# IMPORT CUSTOM CLASSES AND UTILITIES
from custom_classes import (
    PyTorchLSTM, SimpleTokenizer, pad_sequences_manual,
    MAX_WORDS, MAX_SEQ_LENGTH, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, OOV_TOKEN
)

# PyTorch Imports
import torch
import torch.nn as nn


# --- NLTK UTILITIES ---

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Performs tokenization, lowercasing, stopword removal, and stemming/lemmatization."""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# --- 2. LOAD AND PREPROCESS DATA (JIGSAW IMPLEMENTATION) ---

print("Loading and Preprocessing Jigsaw Dataset...")
try:
    df = pd.read_csv('Jigsaw_Toxic_Comment_Dataset.csv')
    # Sampling for faster initial training (You can adjust n= or remove this line)
    # df = df.sample(n=50000, random_state=42) 
    
    TEXT_COLUMN = 'comment_text'
    toxic_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    df['is_toxic'] = df[toxic_labels].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    df = df.dropna(subset=[TEXT_COLUMN, 'is_toxic'])

except FileNotFoundError:
    print("FATAL ERROR: 'jigsaw_toxic_comments.csv' not found. Exiting.")
    exit()

print("Applying NLTK text cleaning and preprocessing...")
df['clean_text'] = df[TEXT_COLUMN].apply(preprocess_text)

X = df['clean_text']
y = df['is_toxic']

# --- 3. SVM MODEL (TF-IDF Feature Extraction and Training) ---

print("Starting SVM Training...")
tfidf_vectorizer = TfidfVectorizer(max_features=MAX_WORDS)
X_tfidf = tfidf_vectorizer.fit_transform(X)
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)
# Use n_jobs=-1 to speed up training using all CPU cores
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train_tfidf, y_train_tfidf)

joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
print("SVM Model and Vectorizer saved.")

# --- 4. PYTORCH LSTM MODEL (Sequential Feature Extraction and Training) ---

print("Starting PyTorch LSTM Training...")

# Use Custom Tokenizer and Padding
lstm_tokenizer = SimpleTokenizer(num_words=MAX_WORDS, oov_token=OOV_TOKEN)
lstm_tokenizer.fit_on_texts(X)

sequences = lstm_tokenizer.texts_to_sequences(X)
X_lstm = pad_sequences_manual(sequences, maxlen=MAX_SEQ_LENGTH)

X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
    X_lstm, y, test_size=0.2, random_state=42
)

X_train_lstm_tensor = torch.LongTensor(X_train_lstm)
y_train_lstm_tensor = torch.FloatTensor(y_train_lstm.values).unsqueeze(1) 

train_data = TensorDataset(X_train_lstm_tensor, y_train_lstm_tensor)
train_loader = DataLoader(train_data, batch_size=32)

vocab_size_actual = len(lstm_tokenizer.word_index) + 1 

pytorch_lstm_model = PyTorchLSTM(
    vocab_size=vocab_size_actual, 
    embedding_dim=EMBEDDING_DIM, 
    hidden_dim=HIDDEN_DIM, 
    output_dim=OUTPUT_DIM, 
    max_seq_length=MAX_SEQ_LENGTH
)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(pytorch_lstm_model.parameters(), lr=0.001)

# Training Loop
pytorch_lstm_model.train()
for epoch in range(5):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = pytorch_lstm_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Save components under the correct module name
torch.save(pytorch_lstm_model.state_dict(), 'pytorch_lstm_model.pt')
joblib.dump(lstm_tokenizer, 'lstm_tokenizer.pkl')
print("PyTorch LSTM Model and Tokenizer saved. Training complete.")