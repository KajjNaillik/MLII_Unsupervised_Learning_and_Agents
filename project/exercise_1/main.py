import numpy as np
import matplotlib.pyplot as plt

# Define the parameters for the joint distribution
mean_age = 30
std_dev_age = 5
mean_height = 160
std_dev_height = 10
covariance = 0.3

# Set a seed for reproducibility
# np.random.seed(42)

# Generate random samples for X (age) and Y (height)
n_samples = 1000
samples = np.random.multivariate_normal(
    mean=[mean_age, mean_height],
    cov=[[std_dev_age**2, covariance], [covariance, std_dev_height**2]],
    size=n_samples
)
print("samples")
print(samples)

# Extract X and Y from the samples
X = samples[:, 0]
Y = samples[:, 1]

# Compute the expected value of Z
expected_value_Z = np.array([np.mean(X), np.mean(Y)])
print("expected_value_Z")
print(expected_value_Z)

# Plot the sampled points in a 2-dimensional figure
plt.scatter(X, Y, alpha=0.5)
plt.title('Sampled Points from Z')
plt.xlabel('Age')
plt.ylabel('Height')
plt.show()

# Compute the empirical average for increasing values of n
empirical_averages = [np.mean(samples[:i+1], axis=0) for i in range(n_samples)]
print("empirical_averages")
print(np.shape(empirical_averages))
# print(empirical_averages)

# Compute Euclidean distances between empirical average and expected value
distances = [np.linalg.norm(empirical_avg - expected_value_Z) for empirical_avg in empirical_averages]
print("distances")
print(np.shape(distances))
# print(distances)

# Plot the Euclidean distance as a function of n
plt.plot(range(1, n_samples + 1), distances)
plt.title('Convergence of Empirical Average to Expected Value')
plt.xlabel('Number of Samples (n)')
plt.ylabel('Euclidean Distance')
plt.show()
