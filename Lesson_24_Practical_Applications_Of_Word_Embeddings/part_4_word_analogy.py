# Perform analogy operation
analogy_result = model.wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
print(analogy_result)
