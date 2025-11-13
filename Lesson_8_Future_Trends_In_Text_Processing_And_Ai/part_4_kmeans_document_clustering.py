from sklearn.cluster import KMeans

# Define the number of clusters
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, random_state=0)

# Fit the model to the TF-IDF matrix
kmeans.fit(tfidf_matrix)

# Predict cluster labels for the documents
clusters = kmeans.labels_
print("Clustered Document Labels:", clusters)
