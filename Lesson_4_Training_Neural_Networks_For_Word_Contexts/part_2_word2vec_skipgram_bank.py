# Training the Word2Vec model using the Skip-Gram technique
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1)
bank_vector = model.wv['bank']
similar_words = model.wv.similar_by_word('bank')
print(f"Vector for 'bank': {bank_vector}")
print("Words similar to 'bank':", similar_words)
