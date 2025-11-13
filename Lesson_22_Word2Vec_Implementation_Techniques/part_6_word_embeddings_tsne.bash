pip install matplotlib scikit-learn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# Extracting word vectors
word_vectors = model.wv[model.wv.key_to_index]

# Using t-SNE for dimensionality reduction
tsne = TSNE(n_components=2)
reduced_vectors = tsne.fit_transform(word_vectors)

# Plotting the results
plt.figure(figsize=(10, 10))
for i, word in enumerate(model.wv.key_to_index):
    plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1])
    plt.annotate(word, xy=(reduced_vectors[i, 0], reduced_vectors[i, 1]))
plt.title("Word Embeddings Visualization")
plt.show()
