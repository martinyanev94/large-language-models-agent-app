specialized_sentences = [
    "Veterinarians provide care for sick animals.",
    "Cats often require regular check-ups.",
    "Pet vaccines are important for health."
]

# Tokenize the specialized sentences
tokenized_specialized_sentences = [word_tokenize(sentence.lower()) for sentence in specialized_sentences]

# Continue training the Word2Vec model
model.build_vocab(tokenized_specialized_sentences, update=True)
model.train(tokenized_specialized_sentences, total_examples=model.corpus_count, epochs=model.epochs)
