import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv(r'C:\Users\Shalini Pal\Documents\2nd Semester\DBSE Project\DataSets\creditcardfraud_normalised.tar.xz')

# Preprocess the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(data_scaled)

# Get the cluster labels and outliers
cluster_labels = dbscan.labels_
outliers_mask = cluster_labels == -1
inliers_mask = cluster_labels != -1

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(data_scaled[inliers_mask, 0], data_scaled[inliers_mask, 1], color='blue', label='Inliers')
plt.scatter(data_scaled[outliers_mask, 0], data_scaled[outliers_mask, 1], color='red', label='Outliers')
plt.legend()
plt.title("DBSCAN Outlier Detection")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Print the outliers
outliers = data[outliers_mask]
print("Outliers detected:")
print(outliers)
