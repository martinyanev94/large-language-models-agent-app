!pip install gensim
import gensim
from gensim.models import Word2Vec

# Sample dataset
sentences = [
    ["the", "king", "lives", "in", "the", "palace"],
    ["the", "queen", "sits", "on", "her", "throne"],
    ["the", "prince", "is", "the", "son", "of", "the", "king"],
    ["the", "princess", "is", "the", "daughter", "of", "the", "queen"],
    ["the", "king", "and", "queen", "rule", "the", "kingdom"]
]

# Training the Word2Vec model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1)
