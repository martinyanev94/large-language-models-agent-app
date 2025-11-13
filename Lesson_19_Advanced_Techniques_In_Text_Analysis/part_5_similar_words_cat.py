similar_words = model.wv.most_similar('cat', topn=5)
print("Words similar to 'cat':", similar_words)
