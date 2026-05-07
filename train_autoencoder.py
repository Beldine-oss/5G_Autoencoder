import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# LOAD AND PREPROCESS DATASET
# ============================================================
def load_dataset():

    print("Loading CSI dataset...")

    data = sio.loadmat("CSI_dataset.mat")
    H = data["H_dataset"]

    print("Original shape:", H.shape)

    # reshape to (samples, antennas, subcarriers)
    H = H.transpose(2,0,1)

    print("Transposed shape:", H.shape)

    # split real and imaginary
    H_real = np.real(H)
    H_imag = np.imag(H)

    # concatenate
    H = np.concatenate([H_real, H_imag], axis=2)

    # flatten
    X = H.reshape(H.shape[0], -1)

    print("Flattened shape:", X.shape)

    # normalize EACH sample (very important)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    return X.astype(np.float32)


# ============================================================
# AUTOENCODER MODEL
# ============================================================
class Autoencoder(nn.Module):

    def __init__(self):

        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(

            nn.Linear(2048, 1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 128)   # compressed representation
        )

        # Decoder
        self.decoder = nn.Sequential(

            nn.Linear(128, 256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.ReLU(),

            nn.Linear(512, 1024),
            nn.ReLU(),

            nn.Linear(1024, 2048)
        )

    def forward(self, x):

        z = self.encoder(x)

        out = self.decoder(z)

        return out


# ============================================================
# TRAINING FUNCTION
# ============================================================
def train():

    X = load_dataset()

    dataset = TensorDataset(torch.tensor(X))

    dataloader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True
    )

    model = Autoencoder()

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-5
    )

    epochs = 100

    loss_history = []

    print("\nStarting Training...\n")

    for epoch in range(epochs):

        total_loss = 0

        for batch in dataloader:

            x = batch[0]

            output = model(x)

            loss = criterion(output, x)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        loss_history.append(avg_loss)

        print(f"Epoch {epoch+1}/{epochs}  Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), "autoencoder_model.pth")

    print("\nModel saved: autoencoder_model.pth")

    plot_loss(loss_history)


# ============================================================
# LOSS PLOT
# ============================================================
def plot_loss(loss):

    plt.figure()

    plt.plot(loss)

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Autoencoder Training Loss")

    plt.grid(True)

    plt.savefig("training_loss.png")

    plt.show()

    print("Saved: training_loss.png")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    train()