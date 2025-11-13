!pip install gensim
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize

# Sample corpi
corpus = [
    "The bank can guarantee deposits will eventually cover future tuition costs because it invests in Adjustable Rate Securities.",
    "The river bank was crowded with tourists.",
    "I need to go to the bank to withdraw some cash.",
    "The bank of the river was formidable."
]

# Tokenizing the sentences
tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in corpus]

# Training the Word2Vec model using Skip-Gram approach
model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, sg=1, min_count=1)

# Checking the word vectors
word_vectors = model.wv
print(word_vectors['bank'])  # Output the vector for the word 'bank'
