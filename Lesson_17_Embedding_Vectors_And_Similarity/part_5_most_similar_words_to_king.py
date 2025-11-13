# Finding the most similar words to 'king'
similar_words = model.most_similar('king', topn=5)
print("Most similar words to 'king':")
for word, similarity in similar_words:
    print(f"{word}: {similarity:.4f}")
