pip install gensim
from gensim.models import FastText

# Sample data: a list of tokenized sentences
sentences = [['i', 'love', 'nlp'],
             ['nlp', 'is', 'amazing'],
             ['i', 'enjoy', 'coding', 'with', 'python'],
             ['let', 'us', 'explore', 'advanced', 'embeddings']]

# Training the FastText model
fasttext_model = FastText(sentences, vector_size=100, window=3, min_count=1, sg=1)

# Getting the vector for the word 'nlp'
nlp_vector = fasttext_model.wv['nlp']
print("Vector for 'nlp':", nlp_vector)
