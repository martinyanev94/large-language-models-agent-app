from keras.layers import LSTM

# Reshape data for LSTM
X_rnn = X.reshape(X.shape[0], X.shape[1], 1)

# Define the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(X_rnn.shape[1], 1)))
lstm_model.add(Flatten())
lstm_model.add(Dense(10, activation='relu'))
lstm_model.add(Dense(1, activation='sigmoid'))

# Compile the LSTM model
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train the LSTM model
lstm_model.fit(X_rnn, y, epochs=10, verbose=1)

# Evaluate the LSTM model
lstm_loss, lstm_accuracy = lstm_model.evaluate(X_rnn, y)
print(f"LSTM Accuracy: {lstm_accuracy * 100:.2f}%")
