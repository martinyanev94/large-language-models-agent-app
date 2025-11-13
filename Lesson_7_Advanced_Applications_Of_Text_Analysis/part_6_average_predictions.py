# Assuming 'cnn_predictions' and 'bert_predictions' contain output probabilities
cnn_predictions = model.predict(data)
bert_predictions = trainer.predict(dataset)

# Averaging predictions
final_predictions = (cnn_predictions + bert_predictions.logits.detach().numpy()) / 2
final_predictions = np.round(final_predictions)  # Rounding to get binary predictions
print(final_predictions)
