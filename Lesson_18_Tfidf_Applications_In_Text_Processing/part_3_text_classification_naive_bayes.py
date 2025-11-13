from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

# Sample documents with categories
documents = [
    "The cat sat on the mat.",
    "The dog sat on the log.",
    "Cats and dogs are great pets.",
    "Dogs love to run more than cats.",
]
categories = ['animals', 'animals', 'pets', 'pets']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(documents, categories, test_size=0.5, random_state=42)

# Create a pipeline combining TF-IDF and the Naive Bayes classifier
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Predict the categories of the test set
predicted_categories = model.predict(X_test)

# Print the classification report
print(classification_report(y_test, predicted_categories))
