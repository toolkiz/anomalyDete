import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv(r'C:\Users\Shalini Pal\Documents\2nd Semester\DBSE Project\DataSets\creditcardfraud_normalised.tar.xz')

# Check for NaN values and impute missing values using the mean
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data)

# Normalize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_imputed)

# Define the autoencoder model
input_dim = data_scaled.shape[1]
encoding_dim = int(input_dim / 2)  # Dimension of the encoding space

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation='relu')(input_layer)
encoder = Dense(int(encoding_dim / 2), activation='relu')(encoder)
decoder = Dense(encoding_dim, activation='relu')(encoder)
decoder = Dense(input_dim, activation='sigmoid')(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder
autoencoder.fit(data_scaled, data_scaled, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)

# Get reconstruction errors
reconstructions = autoencoder.predict(data_scaled)
mse = np.mean(np.power(data_scaled - reconstructions, 2), axis=1)

# Set a threshold for anomaly detection
threshold = np.percentile(mse, 95)  # 95th percentile

# Identify outliers
outliers = mse > threshold
normal_data = mse <= threshold

# Add anomaly column to the original DataFrame
df = pd.DataFrame(data_imputed, columns=data.columns)
df['anomaly'] = outliers

print("Outliers detected:")
print(df[df['anomaly']])

# Visualize the reconstruction error
plt.figure(figsize=(10, 6))
plt.hist(mse[normal_data], bins=50, alpha=0.6, color='blue', label='Normal')
plt.hist(mse[outliers], bins=50, alpha=0.6, color='red', label='Outliers')
plt.axvline(threshold, color='black', linestyle='--', label='Threshold')
plt.legend()
plt.title("Reconstruction Error Histogram")
plt.xlabel("Reconstruction error")
plt.ylabel("Number of samples")
plt.show()
