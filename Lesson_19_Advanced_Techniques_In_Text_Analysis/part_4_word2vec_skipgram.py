# Creating the Word2Vec model using the Skip-gram approach
model_skipgram = Word2Vec(sentences, vector_size=100, window=3, min_count=1, sg=1)

# Checking the vector representation
vector_skipgram = model_skipgram.wv['cat']
print("Vector for 'cat' using Skip-gram:", vector_skipgram)
