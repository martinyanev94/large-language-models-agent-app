import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Embedding
from keras.preprocessing.sequence import pad_sequences

# Sample training data
texts = ["I love programming.", "Python is my favorite language.", "I dislike bugs."]
labels = [1, 1, 0]  # 1 for positive sentiment, 0 for negative

# Convert text to sequences (assuming word_index is generated from Word2Vec)
word_index = {str(i): i for i in range(len(model.wv.key_to_index))}
sequences = [[word_index[word] for word in word_tokenize(text.lower())] for text in texts]
max_sequence_length = max(len(seq) for seq in sequences)
data = pad_sequences(sequences, maxlen=max_sequence_length)

# Building the CNN model
model = Sequential()
model.add(Embedding(input_dim=len(word_index), output_dim=100, input_length=max_sequence_length))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Fitting the model
model.fit(data, np.array(labels), epochs=5)
