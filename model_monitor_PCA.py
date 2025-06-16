import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# === 1. Load Iris and simulate drift ===
X = load_iris().data
X_train, X_test = train_test_split(X, test_size=0.5, random_state=42)

# Inject synthetic drift into first 25 records in test set
X_test[:25, 0] += 2

# === 2. Scale Data ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 3. PCA Train on Normal Data ===
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_train_recon = pca.inverse_transform(X_train_pca)

# === 4. Establish Drift Threshold ===
mse_train = np.mean((X_train_scaled - X_train_recon) ** 2, axis=1)
threshold = np.percentile(mse_train, 95)
print(f"PCA Drift Detection Threshold: {threshold:.4f}")

# === 5. Apply to Incoming Test Data ===
errors = []
drift_flags = []

for i, x in enumerate(X_test_scaled):
    x = x.reshape(1, -1)
    x_pca = pca.transform(x)
    x_recon = pca.inverse_transform(x_pca)
    error = mean_squared_error(x, x_recon)
    errors.append(error)
    drift_flags.append(error > threshold)

# === 6. Plot the Drift Detection ===
plt.figure(figsize=(14, 6))
plt.style.use('seaborn-v0_8-poster')

# Plot all errors
colors = ['#66c2a5' if not flag else '#fc8d62' for flag in drift_flags]
bars = plt.bar(range(len(errors)), errors, color=colors, edgecolor='black')

# Highlight threshold
plt.axhline(y=threshold, color='purple', linestyle='--', linewidth=2, label=f"Threshold = {threshold:.4f}")

# Gradient background
ax = plt.gca()
ax.set_facecolor('#f5f5f5')
ax.grid(True, linestyle='--', alpha=0.6)

# Title and labels
plt.title("PCA Reconstruction Error for Drift Detection", fontsize=18, weight='bold')
plt.xlabel("Sample Index")
plt.ylabel("Reconstruction Error (MSE)")
plt.legend(loc='upper right')

# Annotate drift points
for i, (err, is_drift) in enumerate(zip(errors, drift_flags)):
    if is_drift:
        plt.text(i, err + 0.02, "Drift", ha='center', va='bottom', fontsize=8, color='red', fontweight='bold')

plt.tight_layout()
plt.show()
