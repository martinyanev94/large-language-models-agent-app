def get_analogy(word_a, word_b, word_c):
    return model.wv.most_similar(positive=[word_b, word_c], negative=[word_a])

analogy_result = get_analogy('man', 'woman', 'king')
print(f"Analogy result: {analogy_result}")
