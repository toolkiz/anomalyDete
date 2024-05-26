import numpy as np
import pandas as pd
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv(r'C:\Users\Shalini Pal\Documents\2nd Semester\DBSE Project\DataSets\creditcardfraud_normalised.tar.xz')

# Preprocess the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply OPTICS
optics = OPTICS(min_samples=50, xi=0.05, min_cluster_size=0.1)
optics.fit(data_scaled)

# Extracting the labels (-1 means noise/outlier)
labels = optics.labels_

# Extracting reachability distances
reachability = optics.reachability_[optics.ordering_]

# Identifying outliers
outliers_mask = labels == -1
inliers_mask = labels != -1

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(data_scaled[inliers_mask][:, 0], data_scaled[inliers_mask][:, 1], c='blue', label='Inliers')
plt.scatter(data_scaled[outliers_mask][:, 0], data_scaled[outliers_mask][:, 1], c='red', label='Outliers')
plt.legend()
plt.title("OPTICS Outlier Detection")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Print the outliers
outliers = data[outliers_mask]
print("Outliers detected:")
print(outliers)