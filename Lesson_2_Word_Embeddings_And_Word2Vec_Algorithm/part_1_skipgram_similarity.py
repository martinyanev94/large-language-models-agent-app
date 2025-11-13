# Training with Skip-Gram
model_skipgram = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=2, min_count=1, sg=1)

# Finding similar words
similar_to_king = model_skipgram.wv.similar_by_word('king')
print("Words similar to 'king':", similar_to_king)
