# custom_classes.py (START OF FILE)

import numpy as np
import torch
import torch.nn as nn

# --- Configuration (These must be defined globally for import) ---
MAX_WORDS = 10000
MAX_SEQ_LENGTH = 100  # <--- Ensure this variable is present and defined
EMBEDDING_DIM = 128
HIDDEN_DIM = 64
OUTPUT_DIM = 1
OOV_TOKEN = "<unk>"

# 1. PyTorch LSTM Model Class
class PyTorchLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, max_seq_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden = hidden[-1, :, :] 
        dense_output = self.fc(hidden)
        return self.sigmoid(dense_output)

# 2. Custom Tokenizer Replacement
class SimpleTokenizer:
    def __init__(self, num_words, oov_token="<unk>"):
        self.word_index = {}
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_index[oov_token] = 1 

    def fit_on_texts(self, texts):
        word_counts = {}
        for text in texts:
            for word in text.split():
                word_counts[word] = word_counts.get(word, 0) + 1
        
        sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)
        
        for i, word in enumerate(sorted_words):
            if len(self.word_index) < self.num_words:
                self.word_index[word] = len(self.word_index) + 1
            else:
                break

    def texts_to_sequences(self, texts):
        sequences = []
        oov_index = self.word_index.get(self.oov_token, 1)
        for text in texts:
            sequence = []
            for word in text.split():
                sequence.append(self.word_index.get(word, oov_index))
            sequences.append(sequence)
        return sequences

# 3. Manual Padding Replacement
def pad_sequences_manual(sequences, maxlen, padding='post', truncating='post'):
    """Manual replacement for Keras pad_sequences."""
    padded = np.zeros((len(sequences), maxlen), dtype=np.int32)
    for i, seq in enumerate(sequences):
        if len(seq) > maxlen:
            if truncating == 'post':
                seq = seq[:maxlen]
            else:
                seq = seq[-maxlen:]
        
        if padding == 'post':
            padded[i, :len(seq)] = seq
        else: # 'pre'
            padded[i, maxlen-len(seq):] = seq
    return padded