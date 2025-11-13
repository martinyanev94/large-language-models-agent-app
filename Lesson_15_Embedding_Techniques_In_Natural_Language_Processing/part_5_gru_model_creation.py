from tensorflow.keras.layers import GRU

# Build the GRU model
model_gru = Sequential()
model_gru.add(Embedding(input_dim=10, output_dim=8, input_length=3))
model_gru.add(GRU(16))
model_gru.add(Dense(1, activation='sigmoid'))

# Compile the model
model_gru.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model
model_gru.fit(padded_sequences, labels, epochs=10)
