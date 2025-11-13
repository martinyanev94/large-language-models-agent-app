documents = [
    "FastText is an efficient way to generate word embeddings.",
    "Word embeddings are crucial for various NLP tasks.",
    "Advanced techniques enhance machine understanding of language.",
    "Natural Language Processing includes a wide array of applications.",
    "Cosine similarity measures the resemblance between two vectors."
]

# Create embeddings for the documents using the create_embeddings function
document_embeddings = create_embeddings(documents)
from sklearn.metrics.pairwise import cosine_similarity

def search_query(query):
    query_embedding = create_embeddings([query])[0].reshape(1, -1)
    similarities = cosine_similarity(query_embedding, document_embeddings)
    results = np.argsort(similarities[0])[::-1]  # Sort by similarity in descending order
    return results

# Example query
query = "What is FastText?"
top_results = search_query(query)

print("Top related documents:")
for index in top_results:
    print(documents[index])
