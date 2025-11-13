# Let's say you have more specialized corpus data
specialized_corpus = [
    "Artificial Intelligence is changing the world.",
    "Machine learning algorithms learn from data."
]

# Tokenizing the specialized corpus
tokenized_specialized_corpus = [word_tokenize(sentence.lower()) for sentence in specialized_corpus]

# Continuing the training of the Word2Vec model with new data
model.build_vocab(sentences=tokenized_specialized_corpus, update=True)
model.train(tokenized_specialized_corpus, total_examples=model.corpus_count, epochs=model.epochs)

# Example: Get updated word embedding for 'artificial'
updated_embedding_vector = model.wv['artificial']
print(f"Updated Word Embedding for 'artificial': {updated_embedding_vector}")
