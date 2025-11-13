from glove import Corpus, Glove

# Sample data for GloVe
corpus = Corpus()
corpus.fit([doc.split() for doc in documents], window=5)

# Train GloVe model
glove = Glove(no_components=50, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=100, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)

# Display word vectors
print("\nWord Vectors:")
for word in glove.dictionary:
    print(f"{word}: {glove.word_vectors[glove.dictionary[word]]}")
