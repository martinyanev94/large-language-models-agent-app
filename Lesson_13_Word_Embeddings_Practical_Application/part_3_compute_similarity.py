def compute_similarity(word1, word2):
    return model.wv.similarity(word1, word2)

similarity_score = compute_similarity('word', 'embeddings')
print(f"Similarity between 'word' and 'embeddings': {similarity_score:.2f}")
