from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "Hello, how are you?",
    "I am fine, thank you!",
    "What about you?"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

# Convert to array
X_array = X.toarray()
print(X_array)
print(vectorizer.get_feature_names_out())
