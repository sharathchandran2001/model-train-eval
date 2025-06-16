import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ---------------------------
# 1. Define the Autoencoder
# ---------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)

# ---------------------------
# 2. Simulate training data
# ---------------------------
np.random.seed(0)
X_train = np.random.normal(loc=0, scale=1.0, size=(1000, 10))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)

# ---------------------------
# 3. Train the Autoencoder
# ---------------------------
model = Autoencoder(input_dim=X_train.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train loop
model.train()
for epoch in range(100):
    output = model(X_train_tensor)
    loss = criterion(output, X_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"Training complete. Final loss: {loss.item():.4f}")

# ---------------------------
# 4. Set a Drift Threshold
# ---------------------------
model.eval()
with torch.no_grad():
    recon_train = model(X_train_tensor)
    errors = torch.mean((X_train_tensor - recon_train) ** 2, dim=1)
    threshold = np.percentile(errors.numpy(), 95)  # 95th percentile
print(f"Drift threshold (95th percentile): {threshold:.4f}")

# ---------------------------
# 5. Simulate live data feed
# ---------------------------
# Simulate normal and drifting data
X_live_normal = np.random.normal(loc=0, scale=1.0, size=(100, 10))
X_live_drifted = np.random.normal(loc=2.0, scale=1.0, size=(100, 10))
X_live = np.vstack([X_live_normal, X_live_drifted])

# Scale using training scaler
X_live_scaled = scaler.transform(X_live)
X_live_tensor = torch.tensor(X_live_scaled, dtype=torch.float32)

# ---------------------------
# 6. Monitor for Drift
# ---------------------------
drift_flags = []
errors_live = []

model.eval()
with torch.no_grad():
    for i in range(X_live_tensor.shape[0]):
        x = X_live_tensor[i].unsqueeze(0)
        recon = model(x)
        error = torch.mean((x - recon) ** 2).item()
        errors_live.append(error)
        is_drift = error > threshold
        drift_flags.append(is_drift)
        print(f"[{i}] Error: {error:.4f} {'<-- Drift' if is_drift else ''}")

# ---------------------------
# 7. Plot Results
# ---------------------------
plt.figure(figsize=(12, 4))
plt.plot(errors_live, label="Reconstruction Error")
plt.axhline(y=threshold, color='r', linestyle='--', label="Drift Threshold")
plt.title("Drift Detection with Autoencoder")
plt.xlabel("Data Point")
plt.ylabel("Reconstruction Error")
plt.legend()
plt.show()
