from gensim.models import Word2Vec

# Prepare our sentences
sentences = [
    ["hello", "how", "are", "you"],
    ["i", "am", "fine", "thank", "you"],
    ["what", "about", "you"]
]

# Create the Word2Vec model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1)

# Get the vector for the word 'hello'
vector = model.wv['hello']
print(vector)
