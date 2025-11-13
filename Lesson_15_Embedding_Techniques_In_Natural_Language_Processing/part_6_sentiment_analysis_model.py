from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample dataset
reviews = [
    "I loved this product!",
    "This is the worst purchase I have ever made.",
    "Fairly decent, not the best but good enough."
]
labels = np.array([1, 0, 1])  # 1 for positive, 0 for negative

# Preprocessing the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)
padded_sequences = pad_sequences(sequences)

# Build and train the model
model_sentiment = Sequential()
model_sentiment.add(Embedding(input_dim=100, output_dim=8, input_length=padded_sequences.shape[1]))
model_sentiment.add(GRU(16))
model_sentiment.add(Dense(1, activation='sigmoid'))

model_sentiment.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_sentiment.fit(padded_sequences, labels, epochs=10)
