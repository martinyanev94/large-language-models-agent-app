pip install gensim
from gensim.models import Word2Vec
import nltk
nltk.download('punkt')

# Sample corpus
sentences = [
    "The king rules the kingdom.",
    "The queen has a royal title.",
    "A prince is in line to the throne.",
    "The princess enjoys the royal ball.",
    "Kings and queens often go to war."
]

# Tokenizing the sentences
tokenized_sentences = [nltk.word_tokenize(sentence.lower()) for sentence in sentences]

# Creating the Word2Vec model
model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=2, min_count=1, workers=4)

# Getting the vector representation for the word 'king'
king_vector = model.wv['king']
print("Vector representation of 'king':", king_vector)
