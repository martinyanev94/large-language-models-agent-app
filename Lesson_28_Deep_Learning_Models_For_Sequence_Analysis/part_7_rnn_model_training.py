from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding

# Preparing dummy sequential data for training
data = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
targets = np.array([2, 5, 8])  # Example target outputs

# Creating a basic RNN model
model = Sequential()
model.add(Embedding(input_dim=10, output_dim=4, input_length=3))  # Assume vocab size is 10
model.add(SimpleRNN(10))
model.add(Dense(10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fitting the model 
model.fit(data, targets, epochs=10)
