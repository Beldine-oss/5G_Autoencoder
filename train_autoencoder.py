import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ==================================================
# DATA LOADING (OPTION A FIXED)
# ==================================================
def load_dataset():

    print("Loading dataset...")

    data = sio.loadmat('CSI_dataset.mat')
    H = data['H_dataset']

    print("Original shape:", H.shape)

    # Fix dataset dimensions
    H = H.transpose(2, 0, 1)
    print("Transposed shape:", H.shape)

    # ==================================================
    # OPTION A: KEEP REAL + IMAGINARY PARTS
    # ==================================================
    H_real = np.real(H)
    H_imag = np.imag(H)

    H = np.concatenate([H_real, H_imag], axis=-1)

    print("After complex split shape:", H.shape)

    # Flatten CSI matrices
    X = H.reshape(H.shape[0], -1)
    print("Flattened shape:", X.shape)

    # ==================================================
    # STABLE NORMALIZATION (CRITICAL FIX)
    # ==================================================
    X = (X - np.mean(X)) / (np.std(X) + 1e-8)

    return X


# ==================================================
# AUTOENCODER MODEL (RESEARCH-GRADE)
# ==================================================
class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()

        # =====================
        # ENCODER
        # =====================
        self.encoder = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 128)   # latent space
        )

        # =====================
        # DECODER
        # =====================
        self.decoder = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 2048)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# ==================================================
# TRAINING FUNCTION
# ==================================================
def train_model():

    X = load_dataset()

    X_tensor = torch.tensor(X, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, X_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = Autoencoder()

    criterion = nn.MSELoss()

    # ==================================================
    # STABLE OPTIMIZATION SETTINGS
    # ==================================================
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    epochs = 80

    print("Starting training...")

    loss_history = []

    for epoch in range(epochs):

        total_loss = 0

        for batch_x, _ in loader:

            output = model(batch_x)
            loss = criterion(output, batch_x)

            optimizer.zero_grad()
            loss.backward()

            # gradient stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)

        print(f"Epoch {epoch+1}/{epochs}  Loss: {avg_loss:.6f}")

    # ==================================================
    # SAVE MODEL
    # ==================================================
    torch.save(model.state_dict(), "autoencoder_model.pth")
    print("Training complete. Model saved.")

    # ==================================================
    # LOSS PLOT (FIXED)
    # ==================================================
    plt.figure()
    plt.plot(loss_history)

    plt.title("Autoencoder Training Loss (Option A)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")

    plt.savefig("training_loss.png")
    plt.show()

    print("Training loss graph saved as training_loss.png")


# ==================================================
# ENTRY POINT
# ==================================================
if __name__ == "__main__":
    train_model()