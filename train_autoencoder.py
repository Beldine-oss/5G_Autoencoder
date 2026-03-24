import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


# ===============================
# STEP 1: LOAD CSI DATASET
# ===============================

data = sio.loadmat("channel_dataset.mat")

# MATLAB variable name (change if needed)
H = data["H"]         

print("Original CSI shape:", H.shape)


# ===============================
# STEP 2: PREPARE DATA FOR AI
# ===============================

# Convert complex CSI to real values
H_real = np.real(H)
H_imag = np.imag(H)

# Stack real and imaginary parts
H_combined = np.concatenate((H_real, H_imag), axis=-1)

# Flatten channel matrix for neural network
num_samples = H_combined.shape[0]
input_dim = np.prod(H_combined.shape[1:])

X = H_combined.reshape(num_samples, input_dim)

print("Neural network input shape:", X.shape)

# Convert to float32
X = X.astype(np.float32)

# Convert to torch tensor
X_tensor = torch.from_numpy(X)


# ===============================
# STEP 3: AUTOENCODER MODEL
# ===============================

encoding_dim = 64   # compression size (this represents CSI feedback bits)


class Autoencoder(nn.Module):

    def __init__(self, input_dim, encoding_dim):

        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, encoding_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.ReLU(),

            nn.Linear(512, input_dim)
        )

    def forward(self, x):

        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        return decoded


model = Autoencoder(input_dim, encoding_dim)


# ===============================
# STEP 4: TRAINING SETUP
# ===============================

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50

loss_history = []


# ===============================
# STEP 5: TRAIN AUTOENCODER
# ===============================

for epoch in range(epochs):

    optimizer.zero_grad()

    reconstructed = model(X_tensor)

    loss = criterion(reconstructed, X_tensor)

    loss.backward()

    optimizer.step()

    loss_history.append(loss.item())

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")


# ===============================
# STEP 6: PLOT TRAINING LOSS
# ===============================

plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Autoencoder Training Loss")
plt.show()


# ===============================
# STEP 7: SAVE MODEL
# ===============================

torch.save(model.state_dict(), "csi_autoencoder.pth")

print("Autoencoder model saved successfully.")