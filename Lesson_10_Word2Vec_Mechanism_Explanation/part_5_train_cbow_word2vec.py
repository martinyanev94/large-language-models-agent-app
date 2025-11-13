# Training the CBOW Word2Vec model
model_cbow = Word2Vec(sentences, vector_size=10, window=2, min_count=1, sg=0)
