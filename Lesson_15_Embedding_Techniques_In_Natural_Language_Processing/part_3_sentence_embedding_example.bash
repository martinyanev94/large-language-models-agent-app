pip install sentence-transformers
from sentence_transformers import SentenceTransformer

# Load the pre-trained model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Define sentences
sentences = [
    "The cat sat on the mat.",
    "A dog is lying on the log."
]

# Generate embeddings
sentence_embeddings = model.encode(sentences)

print(sentence_embeddings)
