def compute_similarity(word1, word2):
    return model.wv.similarity(word1, word2)

similarity_score = compute_similarity('fox', 'dog')
print(f"Similarity between 'fox' and 'dog': {similarity_score:.2f}")
