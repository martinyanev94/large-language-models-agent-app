import gensim
from gensim.models import Word2Vec
import nltk
nltk.download('punkt')

# Sample data
sentences = [
    "The cat sat on the mat",
    "The dog sat on the log",
    "Dogs and cats are great pets",
    "The mat is a welcome mat",
]

# Tokenization
tokenized_sentences = [nltk.word_tokenize(sentence.lower()) for sentence in sentences]

# Training the Word2Vec model
model = Word2Vec(tokenized_sentences, vector_size=50, window=2, min_count=1, sg=1)
