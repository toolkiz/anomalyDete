import pandas as pd
import numpy as np
from pyod.models.cof import COF
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv(r'C:\Users\Shalini Pal\Documents\2nd Semester\DBSE Project\DataSets\creditcardfraud_normalised.tar.xz')

# Preprocess the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Reduce dimensions using PCA
pca = PCA(n_components=10)  # Adjust the number of components as needed
data_reduced = pca.fit_transform(data_scaled)

# Apply Connectivity-Based Outlier Factor (COF)
cof = COF()
cof.fit(data_reduced)

# Get outlier scores
outlier_scores = cof.decision_scores_

# Set a threshold for outlier detection (e.g., 95th percentile)
threshold = np.percentile(outlier_scores, 95)

# Identify outliers
outliers_mask = outlier_scores > threshold
inliers_mask = outlier_scores <= threshold

# Visualize the results (for 2D data)
plt.figure(figsize=(10, 6))
plt.scatter(data_reduced[inliers_mask, 0], data_reduced[inliers_mask, 1], color='blue', label='Inliers')
plt.scatter(data_reduced[outliers_mask, 0], data_reduced[outliers_mask, 1], color='red', label='Outliers')
plt.legend()
plt.title("COF Outlier Detection with PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

# Print the outliers
outliers = data[outliers_mask]
print("Outliers detected:")
print(outliers)
