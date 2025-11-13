# Train the model
model.fit(X, y, epochs=10, verbose=1)
# Evaluate the model
loss, accuracy = model.evaluate(X, y)
print(f"Accuracy: {accuracy * 100:.2f}%")
