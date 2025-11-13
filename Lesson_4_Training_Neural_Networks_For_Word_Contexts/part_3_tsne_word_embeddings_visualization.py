from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

words = list(model.wv.key_to_index.keys())
word_vectors = model.wv[words]

# t-SNE
tsne = TSNE(n_components=2)
reduced_vectors = tsne.fit_transform(word_vectors)

plt.figure(figsize=(12, 12))
plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], marker='o')

for i, word in enumerate(words):
    plt.annotate(word, xy=(reduced_vectors[i, 0], reduced_vectors[i, 1]))

plt.title("t-SNE visualization of Word Embeddings")
plt.show()
