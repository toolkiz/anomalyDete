import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

# Example: Create a synthetic dataset
np.random.seed(42)
# Generate normal distributed data
data = np.random.normal(0, 1, 100).reshape(-1, 1)
# Add some outliers
data = np.append(data, [[5], [6], [-5], [-6]]).reshape(-1, 1)

# Convert to DataFrame for better visualization
df = pd.DataFrame(data, columns=['Value'])

# Initialize the Local Outlier Factor model
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)

# Fit the model and predict the outliers
df['anomaly'] = lof.fit_predict(df)

# -1 indicates anomaly, 1 indicates normal
outliers = df[df['anomaly'] == -1]
normal_data = df[df['anomaly'] == 1]

print("Outliers detected:")
print(outliers)

# Visualize the data
plt.figure(figsize=(10, 6))
plt.scatter(normal_data.index, normal_data['Value'], color='blue', label='Normal')
plt.scatter(outliers.index, outliers['Value'], color='red', label='Outliers')
plt.legend()
plt.title("Local Outlier Factor (LOF) Outlier Detection")
plt.xlabel("Index")
plt.ylabel("Value")
plt.show()
