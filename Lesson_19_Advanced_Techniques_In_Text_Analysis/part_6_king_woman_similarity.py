result = model.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=5)
print("King - Man + Woman = ", result)
