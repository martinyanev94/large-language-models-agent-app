vector = model.wv['cat']
print("Vector for 'cat':", vector)
similar_words = model.wv.most_similar('cat')
print("Most similar words to 'cat':", similar_words)
