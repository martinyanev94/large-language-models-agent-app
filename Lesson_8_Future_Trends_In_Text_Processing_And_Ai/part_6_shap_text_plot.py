import shap

# Assuming `model` is our trained text classification model
explainer = shap.Explainer(model)
shap_values = explainer(tfidf_matrix)

# Plotting the SHAP values for the first instance
shap.plots.text(shap_values[0])
