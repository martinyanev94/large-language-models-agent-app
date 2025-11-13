# Finding similar words to "word"
similar_words = model.wv.most_similar('word', topn=5)
print("Words similar to 'word':", similar_words)
