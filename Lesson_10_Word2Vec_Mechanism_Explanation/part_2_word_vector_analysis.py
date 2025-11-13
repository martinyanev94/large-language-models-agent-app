vector_cat = model.wv['cat']
print("Vector for 'cat':", vector_cat)
similar_words = model.wv.most_similar('cat')
print("Most similar words to 'cat':", similar_words)
