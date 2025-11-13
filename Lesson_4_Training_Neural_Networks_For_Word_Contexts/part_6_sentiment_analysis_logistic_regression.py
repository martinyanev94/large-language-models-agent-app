from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Assuming we have sentence embeddings and labels
embeddings = ...  # Represent sentences as their averaged word vectors
labels = ...      # Corresponding sentiments (e.g., positive or negative)

X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2)

# Train logistic regression model
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Evaluate accuracy on test data
accuracy = classifier.score(X_test, y_test)
print(f"Sentiment prediction accuracy: {accuracy}")
