from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Example word vectors
vector_king = model['king'].reshape(1, -1)
vector_queen = model['queen'].reshape(1, -1)

# Calculate cosine similarity
similarity = cosine_similarity(vector_king, vector_queen)
print(f"Cosine Similarity between 'king' and 'queen': {similarity[0][0]}")
