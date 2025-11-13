from sklearn.metrics import classification_report

# Predictions
y_pred = (model.predict(inputs['input_ids']) > 0.5).astype(int)

# Display classification report
print(classification_report(y_test, y_pred))
