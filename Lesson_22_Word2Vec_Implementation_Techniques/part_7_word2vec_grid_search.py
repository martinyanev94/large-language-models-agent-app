from sklearn.model_selection import GridSearchCV

param_grid = {
    'vector_size': [50, 100, 150],
    'window': [2, 5, 10],
    'min_count': [1, 5, 10]
}

# Initialize a Word2Vec model to tune
word2vec_model = Word2Vec(sentences=preprocessed_corpus, sg=1)

# Grid search for the best parameters
clf = GridSearchCV(word2vec_model, param_grid, scoring='accuracy', cv=3)
clf.fit(preprocessed_corpus)
