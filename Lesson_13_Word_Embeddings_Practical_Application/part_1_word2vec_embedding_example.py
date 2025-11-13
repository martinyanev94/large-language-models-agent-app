from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
import numpy as np

nltk.download('punkt')

# Sample corpus
corpus = [
    "Natural language processing is fascinating.",
    "Word embeddings are a key technique in NLP.",
    "Word2Vec allows us to convert words into vectors."
]

# Tokenize the sentences
tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in corpus]

# Create and train the Word2Vec model using the Skip-Gram approach
model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, sg=1)

# Example: Get word embeddings for 'natural'
embedding_vector = model.wv['natural']
print(f"Word Embedding for 'natural': {embedding_vector}")
