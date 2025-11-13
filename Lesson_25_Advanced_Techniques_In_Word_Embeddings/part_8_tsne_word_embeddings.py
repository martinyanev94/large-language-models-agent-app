import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Assume 'embeddings' is a list of word vectors from our trained model
embeddings_2d = TSNE(n_components=2).fit_transform(embeddings)

# Plotting the embeddings
plt.figure(figsize=(10, 10))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
plt.title("t-SNE Visualization of Word Embeddings")
plt.show()
