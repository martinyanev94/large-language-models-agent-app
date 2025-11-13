from sklearn.model_selection import GridSearchCV

# Define parameters for tuning
param_grid = {
    'batch_size': [32, 64],
    'epochs': [5, 10],
    'optimizer': ['adam', 'rmsprop']
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3)
grid.fit(X_train, y_train)

# Best parameters
print("Best parameters found: ", grid.best_params_)
