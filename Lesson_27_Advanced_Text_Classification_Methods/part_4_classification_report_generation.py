from sklearn.metrics import classification_report

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Generate classification report
print(classification_report(y_test, y_pred_classes, target_names=le.classes_))
