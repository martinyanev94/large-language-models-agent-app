from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Convert to array
tfidf_array = tfidf_matrix.toarray()
print(tfidf_array)
print(tfidf_vectorizer.get_feature_names_out())
