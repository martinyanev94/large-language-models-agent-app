from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Generating synthetic high-dimensional data
np.random.seed(0)
data = np.random.rand(100, 10)  # 100 samples, 10 dimensions

# Applying PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)

# Visualizing the reduced data
plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
plt.title('PCA Result - Reduced Dimensions')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
