from gensim.models import Word2Vec
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Generate the Word2Vec model the same way
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1)

# Reduce dimensions to 2D for visualization
words = list(model.wv.index_to_key)
word_vectors = model.wv[words]

pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(word_vectors)

# Visualize the words
plt.figure(figsize=(10, 8))
plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1])

for i, word in enumerate(words):
    plt.annotate(word, xy=(reduced_vectors[i, 0], reduced_vectors[i, 1]), fontsize=12)
plt.grid()
plt.show()
