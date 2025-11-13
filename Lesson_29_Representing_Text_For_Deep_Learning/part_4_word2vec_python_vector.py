from gensim.models import Word2Vec

documents = [
    "I love programming in Python",
    "Python is great for deep learning",
    "I love learning new programming skills"
]

# Preprocess the documents into a list of tokenized words
tokenized_docs = [doc.lower().split() for doc in documents]

model = Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=1, workers=4)

print(model.wv['python'])  # Outputting the vector for the word 'python'
