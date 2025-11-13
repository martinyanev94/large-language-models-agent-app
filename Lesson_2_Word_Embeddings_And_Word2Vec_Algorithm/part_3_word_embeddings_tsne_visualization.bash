pip install matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Retrieve word vectors
words = list(model_skipgram.wv.index_to_key)
word_vectors = model_skipgram.wv[words]

# Executing T-SNE
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(word_vectors)

# Plotting the results
plt.figure(figsize=(15, 10))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])

for i, word in enumerate(words):
    plt.annotate(word, xy=(X_tsne[i, 0], X_tsne[i, 1]))

plt.title("Word Embeddings Visualization (T-SNE)")
plt.xlabel("TSNE Component 1")
plt.ylabel("TSNE Component 2")
plt.grid(True)
plt.show()
