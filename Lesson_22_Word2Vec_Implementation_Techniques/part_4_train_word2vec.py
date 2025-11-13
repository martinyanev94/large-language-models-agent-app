from gensim.models import Word2Vec

# Training the Word2Vec model
model = Word2Vec(sentences=preprocessed_corpus, vector_size=100, window=5, min_count=1, sg=1)
