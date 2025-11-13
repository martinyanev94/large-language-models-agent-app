pip install tensorflow
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# Sample dataset
sequences = [
    "I love programming in Python.",
    "Python is an amazing language.",
    "I enjoy building AI applications."
]

# Preprocess the text - tokenization and numerical encoding
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(sequences)
sequences_encoded = tokenizer.texts_to_sequences(sequences)

# Padding sequences to have uniform length
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences_encoded)

# Define RNN model
model = Sequential()
model.add(Embedding(input_dim=10, output_dim=8, input_length=padded_sequences.shape[1]))
model.add(SimpleRNN(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("RNN model summary:")
model.summary()
