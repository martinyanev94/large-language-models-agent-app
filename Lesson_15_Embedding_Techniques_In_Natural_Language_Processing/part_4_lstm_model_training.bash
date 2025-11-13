pip install tensorflow
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Prepare sample data
sentences = [[1, 2, 3], [4, 5, 6], [1, 5, 2]]
labels = np.array([0, 1, 0])  # Binary labels

# Pad sequences to ensure uniform input size
padded_sequences = pad_sequences(sentences, padding='post')

# Build the LSTM model
model = Sequential()
model.add(Embedding(input_dim=10, output_dim=8, input_length=3))
model.add(LSTM(16))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(padded_sequences, labels, epochs=10)
