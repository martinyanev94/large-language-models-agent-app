from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = [
    "The cat sat on the mat.",
    "Dogs are great pets.",
    "Cats and dogs are wonderful companions.",
    "The quick brown fox jumps over the lazy dog."
]

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the documents to get the TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Display the TF-IDF scores
tfidf_array = tfidf_matrix.toarray()
print("TF-IDF Scores:\n", tfidf_array)

# Get feature names for insights
feature_names = tfidf_vectorizer.get_feature_names_out()
for doc_idx, doc in enumerate(tfidf_array):
    print(f"\nDocument {doc_idx + 1}:")
    for word_idx, score in enumerate(doc):
        if score > 0:
            print(f"{feature_names[word_idx]}: {score:.4f}")
