import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------
# Load dataset
# -----------------------------
print("Loading dataset...")

data = sio.loadmat('CSI_dataset.mat')
H = data['H_dataset']

print("Original shape:", H.shape)

# -----------------------------
# Fix dataset dimensions
# (samples, antennas)
# -----------------------------
H = H.transpose(2,0,1)
print("Transposed shape:", H.shape)

# -----------------------------
# Flatten CSI matrices
# -----------------------------
X = H.reshape(H.shape[0], -1)
print("Flattened shape:", X.shape)

# -----------------------------
# Normalize data
# -----------------------------
X = X / np.max(np.abs(X))

# -----------------------------
# Convert to PyTorch tensor
# -----------------------------
X_tensor = torch.tensor(X, dtype=torch.float32)

dataset = TensorDataset(X_tensor, X_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# -----------------------------
# Define Autoencoder
# -----------------------------
class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )

        self.decoder = nn.Sequential(
            nn.Linear(32,128),
            nn.ReLU(),
            nn.Linear(128,512),
            nn.ReLU(),
            nn.Linear(512,1024)
        )

    def forward(self, x):

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


model = Autoencoder()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# Training loop
# -----------------------------
print("Starting training...")

epochs = 20
loss_history = []

for epoch in range(epochs):

    total_loss = 0

    for batch_x, _ in loader:

        output = model(batch_x)

        loss = criterion(output, batch_x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    loss_history.append(avg_loss)

    print(f"Epoch {epoch+1}/{epochs}  Loss: {avg_loss:.6f}")

# -----------------------------
# Save trained model
# -----------------------------

torch.save(model.state_dict(), "autoencoder_model.pth")

print("Training complete. Model saved.")
# -----------------------------
# Plot Training Loss
# -----------------------------

import matplotlib.pyplot as plt

plt.figure()

plt.plot(loss_history)

plt.title("Autoencoder Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")

plt.show()
plt.savefig("training_loss.png")


print("Training loss graph saved as training_loss.png")

