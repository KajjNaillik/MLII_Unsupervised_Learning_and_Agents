import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt

data = np.load('data.npy')

def elbow_method(data, max_clusters=10):
    distortions = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_init=10, n_clusters=i, random_state=42)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)
    return distortions

def silhouette_method(data, max_clusters=10):
    silhouette_scores = []
    for i in range(2, max_clusters + 1):
        kmeans = KMeans(n_init=10, n_clusters=i, random_state=42)
        labels = kmeans.fit_predict(data)
        silhouette_scores.append(silhouette_score(data, labels))
    return silhouette_scores

def elbow_method_hie(data, max_clusters=10):
    distortions = []
    for i in range(1, max_clusters + 1):
        agglomerative = AgglomerativeClustering(n_clusters=i, metric='manhattan', linkage='complete')
        labels = agglomerative.fit_predict(data)
        variance = sum(np.sum(np.var(data[np.where(labels == j)], axis=0)) for j in np.unique(labels))
        distortions.append(variance)
    return distortions

def silhouette_method_hie(data, max_clusters=10):
    silhouette_scores = []
    for i in range(2, max_clusters + 1):
        hierarchical = AgglomerativeClustering(n_clusters=i, metric='manhattan', linkage='complete')
        labels = hierarchical.fit_predict(data)
        score = silhouette_score(data, labels)
        silhouette_scores.append(score)
    return silhouette_scores

methods = ['K-Means', 'Hierarchical']

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

distortions = elbow_method(data)
axes[0, 0].plot(range(1, len(distortions) + 1), distortions, marker='o')
axes[0, 0].set_title('K-Means - Elbow Method')
axes[0, 0].set_xlabel('Number of Clusters')
axes[0, 0].set_ylabel('Distortion')

silhouette_scores = silhouette_method(data)
axes[0, 1].plot(range(2, len(silhouette_scores) + 2), silhouette_scores, marker='o')
axes[0, 1].set_title('K-Means - Silhouette Method')
axes[0, 1].set_xlabel('Number of Clusters')
axes[0, 1].set_ylabel('Silhouette Score')

distortions = elbow_method_hie(data)
axes[1, 0].plot(range(1, len(distortions) + 1), distortions, marker='o')
axes[1, 0].set_title('Hierarchical - Elbow Method')
axes[1, 0].set_xlabel('Number of Clusters')
axes[1, 0].set_ylabel('Distortion')

silhouette_scores = silhouette_method_hie(data)
axes[1, 1].plot(range(2, len(silhouette_scores) + 2), silhouette_scores, marker='o')
axes[1, 1].set_title('Hierarchical - Silhouette Method')
axes[1, 1].set_xlabel('Number of Clusters')
axes[1, 1].set_ylabel('Silhouette Score')

plt.tight_layout()
plt.show()
