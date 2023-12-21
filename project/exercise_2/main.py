import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load data and labels
data = np.load('exercise_2/data.npy')
labels = np.load('exercise_2/labels.npy')

# Function to plot 2D and 3D scatter plots
def plot_scatter(data_2d, data_3d, labels):
    print("labels")
    print(np.shape(labels))

    # Plot 2D scatter plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.title('2D Dimensionality Reduction')
    print("data_2d")
    print(np.shape(data_2d))

    # Plot 3D scatter plot
    ax = plt.subplot(1, 2, 2, projection='3d')
    ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=labels, cmap='viridis', alpha=0.5)
    ax.set_title('3D Dimensionality Reduction')
    print("data_3d")
    print(np.shape(data_3d))

    plt.show()

# Perform PCA for dimensionality reduction to 2D and 3D
pca_2d = PCA(n_components=2)
pca_3d = PCA(n_components=3)

data_pca_2d = pca_2d.fit_transform(data)
data_pca_3d = pca_3d.fit_transform(data)

# Visualize PCA results
plot_scatter(data_pca_2d, data_pca_3d, labels)

# Perform t-SNE for dimensionality reduction to 2D and 3D
tsne_2d = TSNE(n_components=2)
tsne_3d = TSNE(n_components=3)

data_tsne_2d = tsne_2d.fit_transform(data)
data_tsne_3d = tsne_3d.fit_transform(data)

# Visualize t-SNE results
plot_scatter(data_tsne_2d, data_tsne_3d, labels)
