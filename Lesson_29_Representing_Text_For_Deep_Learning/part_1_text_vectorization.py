from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "I love programming in Python",
    "Python is great for deep learning",
    "I love learning new programming skills"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())
print(X.toarray())
