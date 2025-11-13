from tensorflow.keras.layers import LSTM

# Creating the LSTM model
model_lstm = Sequential()
model_lstm.add(Embedding(input_dim=10, output_dim=4, input_length=3))
model_lstm.add(LSTM(10))
model_lstm.add(Dense(10, activation='softmax'))

model_lstm.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fitting the model 
model_lstm.fit(data, targets, epochs=10)
