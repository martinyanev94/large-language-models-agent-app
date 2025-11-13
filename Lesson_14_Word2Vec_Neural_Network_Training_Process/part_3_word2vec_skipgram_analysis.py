from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK punkt tokenizer
nltk.download('punkt')

# Sample sentences
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "The dog barked at the fox.",
    "The lazy dog sleeps all day."
]

# Tokenize sentences
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

# Create a Word2Vec model using Skip-Gram
model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=2, min_count=1, sg=1)
