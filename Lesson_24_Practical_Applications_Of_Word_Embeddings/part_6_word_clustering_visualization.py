import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Get the word embeddings and list of words
words = list(model.wv.key_to_index)
word_vectors = model.wv[words]

# Cluster the embeddings
kmeans = KMeans(n_clusters=5)
kmeans.fit(word_vectors)
clusters = kmeans.predict(word_vectors)

# Plotting the clusters
plt.figure(figsize=(12, 8))
for i, word in enumerate(words):
    plt.scatter(word_vectors[i][0], word_vectors[i][1], c=clusters[i], label=word)

plt.title('Word Clustering Visualization')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend(loc='best')
plt.show()
