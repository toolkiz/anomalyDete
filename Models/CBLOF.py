import pandas as pd
import numpy as np
from pyod.models.cblof import CBLOF
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv(r'C:\Users\Shalini Pal\Documents\2nd Semester\DBSE Project\DataSets\creditcardfraud_normalised.tar.xz')

# Preprocess the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply Cluster-Based Local Outlier Factor (CBLOF)
cblof = CBLOF()
cblof.fit(data_scaled)

# Get outlier scores
outlier_scores = cblof.decision_scores_

# Set a threshold for outlier detection (e.g., 95th percentile)
threshold = np.percentile(outlier_scores, 95)

# Identify outliers
outliers_mask = outlier_scores > threshold
inliers_mask = outlier_scores <= threshold

# Visualize the results (for 2D data)
plt.figure(figsize=(10, 6))
plt.scatter(data_scaled[inliers_mask, 0], data_scaled[inliers_mask, 1], color='blue', label='Inliers')
plt.scatter(data_scaled[outliers_mask, 0], data_scaled[outliers_mask, 1], color='red', label='Outliers')
plt.legend()
plt.title("CBLOF Outlier Detection")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Print the outliers
outliers = data[outliers_mask]
print("Outliers detected:")
print(outliers)
