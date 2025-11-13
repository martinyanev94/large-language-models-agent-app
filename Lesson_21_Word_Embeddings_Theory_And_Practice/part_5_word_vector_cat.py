# Getting the vector for the word 'cat'
cat_vector = model.wv['cat']
print("Vector for 'cat':", cat_vector)

# Finding similar words to 'cat'
similar_words = model.wv.most_similar('cat', topn=5)
print("Words similar to 'cat':", similar_words)
