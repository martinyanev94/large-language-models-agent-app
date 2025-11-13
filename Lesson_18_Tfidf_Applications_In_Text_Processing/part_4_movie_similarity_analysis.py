from sklearn.metrics.pairwise import cosine_similarity

# Sample movie descriptions
movie_descriptions = [
    "A thrilling adventure in space with a brave hero.",
    "A romantic drama set in a bustling city.",
    "A gripping tale of survival against the odds.",
    "An exciting story of friendship and love.",
]

# Transforming the movie descriptions into TF-IDF
tfidf_matrix_movies = vectorizer.fit_transform(movie_descriptions)

# Calculate cosine similarity between the first movie and the rest
similarities = cosine_similarity(tfidf_matrix_movies[0:1], tfidf_matrix_movies).flatten()

# Get the index of the most similar movie
similar_movies_idx = similarities.argsort()[1:]

print("Similar movies to the first description:")
for idx in similar_movies_idx:
    print(f"Movie: {movie_descriptions[idx]}, Similarity Score: {similarities[idx]:.4f}")
