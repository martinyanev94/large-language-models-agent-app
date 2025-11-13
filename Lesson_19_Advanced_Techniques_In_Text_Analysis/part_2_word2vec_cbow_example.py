from gensim.models import Word2Vec
import numpy as np

# Sample sentences
sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["dogs", "are", "better", "than", "cats"],
    ["cats", "are", "better", "than", "mice"]
]

# Creating the Word2Vec model using the CBOW approach
model = Word2Vec(sentences, vector_size=100, window=3, min_count=1, sg=0)

# Checking the vector representation
vector = model.wv['cat']
print("Vector for 'cat':", vector)
