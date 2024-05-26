import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv(r'C:\Users\Shalini Pal\Documents\2nd Semester\DBSE Project\DataSets\creditcardfraud_normalised.tar.xz')

# Preprocess the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(data_scaled)

# Get the probability densities for each data point
densities = gmm.score_samples(data_scaled)
threshold = np.percentile(densities, 5)  # Set a threshold (e.g., 5th percentile)

# Identify outliers
outliers_mask = densities < threshold
inliers_mask = densities >= threshold

# Visualize the results (for 2D data)
plt.figure(figsize=(10, 6))
plt.scatter(data_scaled[inliers_mask, 0], data_scaled[inliers_mask, 1], color='blue', label='Inliers')
plt.scatter(data_scaled[outliers_mask, 0], data_scaled[outliers_mask, 1], color='red', label='Outliers')
plt.legend()
plt.title("GMM Outlier Detection")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Print the outliers
outliers = data[outliers_mask]
print("Outliers detected:")
print(outliers)
