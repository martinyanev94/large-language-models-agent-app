# Example: Performing analogy on word embeddings
def get_analogy(word_a, word_b, word_c):
    result = model.wv.most_similar(positive=[word_b, word_c], negative=[word_a])
    return result

analogy_result = get_analogy('man', 'woman', 'king')
print(f"Analogy Result: {analogy_result}")
