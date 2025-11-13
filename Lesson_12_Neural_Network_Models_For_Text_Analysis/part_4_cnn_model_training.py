from keras.layers import Conv1D, MaxPooling1D, Flatten

# Reshape data for Conv1D
X_cnn = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape to (samples, time steps, features)

# Define the CNN model
cnn_model = Sequential()
cnn_model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(X_cnn.shape[1], 1)))
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Flatten())
cnn_model.add(Dense(10, activation='relu'))
cnn_model.add(Dense(1, activation='sigmoid'))

# Compile the model
cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train the CNN model
cnn_model.fit(X_cnn, y, epochs=10, verbose=1)

# Evaluate the CNN model
cnn_loss, cnn_accuracy = cnn_model.evaluate(X_cnn, y)
print(f"CNN Accuracy: {cnn_accuracy * 100:.2f}%")
