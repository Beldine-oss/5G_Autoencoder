import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    print("Loading dataset...")

    data = sio.loadmat("CSI_dataset.mat")

    H = data["H_dataset"]

    print("Original shape:", H.shape)

    # (16,64,10000) -> (10000,16,64)
    H = H.transpose(2,0,1)

    print("Transposed shape:", H.shape)

    # Separate real and imaginary
    H_real = np.real(H)
    H_imag = np.imag(H)

    # Shape -> (samples,2,16,64)
    X = np.stack([H_real, H_imag], axis=1)

    print("CNN input shape:", X.shape)

    # Global scaling
    X_max = np.max(np.abs(X))

    X = X / X_max

    return X.astype(np.float32)


# ============================================================
# RESIDUAL BLOCK
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(self, channels):

        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(channels),

            nn.LeakyReLU(0.3),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(channels)
        )

        self.activation = nn.LeakyReLU(0.3)

    def forward(self, x):

        residual = x

        out = self.block(x)

        out = out + residual

        out = self.activation(out)

        return out


# ============================================================
# CNN AUTOENCODER
# ============================================================

class CNNAutoencoder(nn.Module):

    def __init__(self):

        super(CNNAutoencoder, self).__init__()

        # ====================================================
        # ENCODER
        # ====================================================

        self.encoder = nn.Sequential(

            nn.Conv2d(2, 16, kernel_size=3, padding=1),

            nn.BatchNorm2d(16),

            nn.LeakyReLU(0.3),

            ResidualBlock(16),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),

            nn.BatchNorm2d(32),

            nn.LeakyReLU(0.3),

            ResidualBlock(32),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),

            nn.BatchNorm2d(64),

            nn.LeakyReLU(0.3),

            ResidualBlock(64)
        )

        # ====================================================
        # BOTTLENECK
        # ====================================================

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(64 * 4 * 16, 256)

        self.fc2 = nn.Linear(256, 64 * 4 * 16)

        # ====================================================
        # DECODER
        # ====================================================

        self.decoder = nn.Sequential(

            ResidualBlock(64),

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.LeakyReLU(0.3),

            ResidualBlock(32),

            nn.ConvTranspose2d(
                32,
                16,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(16),

            nn.LeakyReLU(0.3),

            ResidualBlock(16),

            nn.Conv2d(
                16,
                2,
                kernel_size=3,
                padding=1
            ),

            nn.Tanh()
        )

    def forward(self, x):

        x = self.encoder(x)

        x = self.flatten(x)

        encoded = self.fc1(x)

        x = self.fc2(encoded)

        x = x.view(-1, 64, 4, 16)

        decoded = self.decoder(x)

        return decoded


# ============================================================
# HYBRID LOSS
# ============================================================

def hybrid_loss(output, target):

    # Standard MSE
    mse = F.mse_loss(output, target)

    # Flatten
    output_flat = output.view(output.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    # Cosine similarity
    cosine = 1 - F.cosine_similarity(
        output_flat,
        target_flat,
        dim=1
    ).mean()

    # Combined loss
    loss = mse + 0.2 * cosine

    return loss


# ============================================================
# NMSE
# ============================================================

def calculate_nmse(actual, predicted):

    mse = torch.sum((actual - predicted) ** 2)

    power = torch.sum(actual ** 2)

    nmse = mse / (power + 1e-9)

    return nmse


# ============================================================
# TRAINING
# ============================================================

def train_model():

    X = load_dataset()

    X_tensor = torch.tensor(X)

    dataset = TensorDataset(X_tensor, X_tensor)

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True
    )

    model = CNNAutoencoder().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10
    )

    epochs = 250

    loss_history = []
    nmse_history = []

    print("\nStarting training...\n")

    for epoch in range(epochs):

        model.train()

        total_loss = 0
        total_nmse = 0

        for batch_x, _ in loader:

            batch_x = batch_x.to(device)

            output = model(batch_x)

            loss = hybrid_loss(output, batch_x)

            nmse = calculate_nmse(batch_x, output)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            total_nmse += nmse.item()

        avg_loss = total_loss / len(loader)
        avg_nmse = total_nmse / len(loader)

        scheduler.step(avg_loss)

        loss_history.append(avg_loss)
        nmse_history.append(avg_nmse)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {avg_loss:.6f} | "
            f"NMSE: {avg_nmse:.6f}"
        )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    torch.save(
        model.state_dict(),
        "autoencoder_model.pth"
    )

    print("\nTraining complete.")

    # ========================================================
    # PLOT
    # ========================================================

    plt.figure(figsize=(10,5))

    plt.plot(loss_history, label="Loss")
    plt.plot(nmse_history, label="NMSE")

    plt.xlabel("Epoch")

    plt.ylabel("Value")

    plt.title("Training Performance")

    plt.legend()

    plt.grid(True)

    plt.savefig("training_performance.png")

    plt.show()

    print("Saved: training_performance.png")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    train_model()