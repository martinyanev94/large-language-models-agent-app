import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

# Example parameters
vocab_size = 1000  # Assume a vocabulary size of 1000
embedding_dim = 128  # Dimension of the embedding vector
max_length = 10  # Max length of input sequences

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
model.add(LSTM(100))  # LSTM layer with 100 units
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
