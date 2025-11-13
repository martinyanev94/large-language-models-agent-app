from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "I love programming in Python",
    "Python is great for deep learning",
    "I love learning new programming skills"
]

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

print(tfidf_vectorizer.get_feature_names_out())
print(tfidf_matrix.toarray())
