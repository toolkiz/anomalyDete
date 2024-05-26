import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv(r'C:\Users\Shalini Pal\Documents\2nd Semester\DBSE Project\DataSets\creditcardfraud_normalised.tar.xz')

# Preprocess the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply k-means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(data_scaled)

# Get cluster centers and calculate distances to centers for each point
cluster_centers = kmeans.cluster_centers_
distances = np.sqrt(((data_scaled - cluster_centers[:, np.newaxis])**2).sum(axis=2))

# Get the maximum distance for each point
max_distances = np.max(distances, axis=0)

# Set a threshold for outlier detection (e.g., 95th percentile)
threshold = np.percentile(max_distances, 95)

# Identify outliers
outliers_mask = max_distances > threshold
inliers_mask = max_distances <= threshold

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(data_scaled[inliers_mask, 0], data_scaled[inliers_mask, 1], color='blue', label='Inliers')
plt.scatter(data_scaled[outliers_mask, 0], data_scaled[outliers_mask, 1], color='red', label='Outliers')
plt.legend()
plt.title("K-Means Outlier Detection")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Print the outliers
outliers = data[outliers_mask]
print("Outliers detected:")
print(outliers)
