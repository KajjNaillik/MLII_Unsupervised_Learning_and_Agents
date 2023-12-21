import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt

# Load the data
data = np.load('./data.npy')

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Define clustering methods
kmeans = KMeans(n_clusters=3, random_state=42)
hierarchical = AgglomerativeClustering(n_clusters=3)

# Define heuristics
def elbow_method(data, max_clusters=10):
    distortions = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)
    return distortions

def silhouette_method(data, max_clusters=10):
    silhouette_scores = []
    for i in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        labels = kmeans.fit_predict(data)
        silhouette_scores.append(silhouette_score(data, labels))
    return silhouette_scores

# Apply clustering methods and heuristics
methods = [(kmeans, 'K-Means', 'Euclidean'), (hierarchical, 'Hierarchical', 'Euclidean')]

for method, method_name, metric_name in methods:
    # Elbow method
    distortions = elbow_method(data)
    plt.plot(range(1, len(distortions) + 1), distortions, marker='o', label=f'{method_name} - Elbow')

    # Silhouette method
    silhouette_scores = silhouette_method(data)
    plt.plot(range(2, len(silhouette_scores) + 2), silhouette_scores, marker='o', label=f'{method_name} - Silhouette')

    # Fit and predict
    labels = method.fit_predict(data)

    # Metric for evaluation
    if metric_name == 'Euclidean':
        metric = 'euclidean'
    else:
        # Define your custom metric here
        pass

    # Print the evaluation metric
    print(f'{method_name} - {metric_name} Metric:', calinski_harabasz_score(data, labels))

# Plotting
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.legend()
plt.show()
