from sklearn.feature_extraction.text import TfidfVectorizer

# Creating the TfidfVectorizer instance
tfidf_vectorizer = TfidfVectorizer()

# Fitting and transforming the documents
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Convert TF-IDF matrix to array
tfidf_array = tfidf_matrix.toarray()
print("TF-IDF representation:\n", tfidf_array)
print("Feature names (words):", tfidf_vectorizer.get_feature_names_out())
