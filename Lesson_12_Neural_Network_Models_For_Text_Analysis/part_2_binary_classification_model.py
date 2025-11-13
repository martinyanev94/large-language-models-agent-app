# Define the model
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=X.shape[1]))  # Input layer with ReLU activation
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
