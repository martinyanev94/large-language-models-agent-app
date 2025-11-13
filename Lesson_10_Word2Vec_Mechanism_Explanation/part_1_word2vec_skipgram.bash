pip install gensim
from gensim.models import Word2Vec

# Sample sentences
sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "sat", "on", "the", "log"],
    ["the", "cat", "and", "the", "dog", "are", "friends"],
    ["cats", "and", "dogs", "are", "common", "pets"]
]

# Training the Skip-Gram Word2Vec model
model = Word2Vec(sentences, vector_size=10, window=2, min_count=1, sg=1)
