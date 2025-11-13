king_vector = model.wv['king']
queen_vector = model.wv['queen']
prince_vector = model.wv['prince']
princess_vector = model.wv['princess']

print(f"King vector: {king_vector}")
print(f"Queen vector: {queen_vector}")
print(f"Prince vector: {prince_vector}")
print(f"Princess vector: {princess_vector}")
from numpy import dot
from numpy.linalg import norm

def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

similarity = cosine_similarity(king_vector, queen_vector)
print(f"Cosine similarity between 'king' and 'queen': {similarity}")
