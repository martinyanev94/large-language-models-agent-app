pip install gensim
from gensim.models import Word2Vec
import nltk
nltk.download('punkt')

# Sample text data
text_data = [
    "Language is an incredible tool for communication.",
    "Natural Language Processing allows AI to understand human language.",
    "Word embeddings are a way to represent words in vector space."
]

# Tokenization
tokens = [nltk.word_tokenize(sentence.lower()) for sentence in text_data]

# Create Word2Vec model
model = Word2Vec(sentences=tokens, vector_size=10, window=2, min_count=1, workers=4)

# Getting the vector representation of a word
word_vector = model.wv['language']
print("Vector representation of 'language':", word_vector)
