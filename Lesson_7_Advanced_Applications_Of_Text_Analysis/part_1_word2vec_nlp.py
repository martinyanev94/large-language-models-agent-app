from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK resources
nltk.download('punkt')

# Sample corpus
corpus = [
    "Natural language processing is a fascinating area of AI.",
    "Deep learning techniques can greatly enhance text analysis.",
    "Word embeddings are crucial for understanding the semantics of text."
]

# Tokenizing sentences into words
tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in corpus]

# Creating a Word2Vec model
model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)

# Accessing vector for a word
word_vector = model.wv['deep']
print(f"Vector for 'deep': {word_vector}")
