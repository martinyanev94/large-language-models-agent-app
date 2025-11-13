pip install scikit-learn
from sklearn.feature_extraction.text import CountVectorizer

# Sample documents
documents = [
    "Language is an incredible tool for communication.",
    "Natural Language Processing allows AI to understand human language.",
    "Word embeddings are a way to represent words in vector space."
]

# Creating the CountVectorizer instance
vectorizer = CountVectorizer()

# Fitting and transforming the documents
bow_matrix = vectorizer.fit_transform(documents)

# Convert BoW matrix to array
bow_array = bow_matrix.toarray()
print("Bag of Words representation:\n", bow_array)
print("Feature names (words):", vectorizer.get_feature_names_out())
