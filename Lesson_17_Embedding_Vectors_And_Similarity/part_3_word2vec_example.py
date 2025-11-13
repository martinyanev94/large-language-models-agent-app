from gensim.models import Word2Vec
from gensim.models import KeyedVectors

# Load a pre-trained Word2Vec model
# For the sake of example, we will use Google News vectors
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Get the vector for a word
word_vector = model['king']
print(word_vector)
