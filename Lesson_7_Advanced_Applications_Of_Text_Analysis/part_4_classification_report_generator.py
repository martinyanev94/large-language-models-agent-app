from sklearn.metrics import classification_report

# Simulating predictions for evaluation
predictions = model.predict_classes(data)
print(classification_report(labels, predictions))
