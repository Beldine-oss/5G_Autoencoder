import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# DEVICE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA Available:", torch.cuda.is_available())
print("Using device:", device)

# ============================================================
# DATA LOADING
# ============================================================
def load_dataset():

    print("Loading dataset...")

    data = sio.loadmat("CSI_dataset.mat")

    H = data["H_dataset"]

    print("Original shape:", H.shape)

    # (16,64,10000) -> (10000,16,64)
    H = H.transpose(2, 0, 1)

    print("Transposed shape:", H.shape)

    # --------------------------------------------------------
    # REAL + IMAGINARY
    # --------------------------------------------------------
    H_real = np.real(H)

    H_imag = np.imag(H)

    # --------------------------------------------------------
    # STACK CHANNELS
    # Shape: (10000,2,16,64)
    # --------------------------------------------------------
    X = np.stack([H_real, H_imag], axis=1)

    print("CNN Input shape:", X.shape)

    # --------------------------------------------------------
    # GLOBAL MAX NORMALIZATION
    # --------------------------------------------------------
    X_max = np.max(np.abs(X))

    X = X / X_max

    print("Normalization complete.")

    return X.astype(np.float32)


# ============================================================
# RESIDUAL BLOCK
# ============================================================
class ResidualBlock(nn.Module):

    def __init__(self, channels):

        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(channels)

        self.activation = nn.LeakyReLU(0.3)

    def forward(self, x):

        residual = x

        out = self.conv1(x)

        out = self.bn1(out)

        out = self.activation(out)

        out = self.conv2(out)

        out = self.bn2(out)

        out = out + residual

        out = self.activation(out)

        return out


# ============================================================
# LIGHTWEIGHT RESIDUAL CNN AUTOENCODER
# ============================================================
class ResidualCsiAutoencoder(nn.Module):

    def __init__(self):

        super(ResidualCsiAutoencoder, self).__init__()

        # ====================================================
        # ENCODER
        # ====================================================

        self.encoder = nn.Sequential(

            # INPUT: (2,16,64)

            nn.Conv2d(
                in_channels=2,
                out_channels=4,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(4),

            nn.LeakyReLU(0.3),

            ResidualBlock(4),

            # 16x64 -> 8x32
            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=4,
                out_channels=8,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(8),

            nn.LeakyReLU(0.3),

            # 8x32 -> 4x16
            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(16),

            nn.LeakyReLU(0.3)
        )

        # ====================================================
        # FLATTEN
        # ====================================================

        self.flatten = nn.Flatten()

        # 16 × 4 × 16 = 1024

        # ====================================================
        # BOTTLENECK
        # ====================================================

        self.encoder_fc = nn.Linear(1024, 256)

        self.decoder_fc = nn.Linear(256, 1024)

        # ====================================================
        # DECODER
        # ====================================================

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=8,
                kernel_size=2,
                stride=2
            ),

            nn.BatchNorm2d(8),

            nn.LeakyReLU(0.3),

            nn.ConvTranspose2d(
                in_channels=8,
                out_channels=4,
                kernel_size=2,
                stride=2
            ),

            nn.BatchNorm2d(4),

            nn.LeakyReLU(0.3),

            ResidualBlock(4),

            nn.Conv2d(
                in_channels=4,
                out_channels=2,
                kernel_size=3,
                padding=1
            ),

            nn.Tanh()
        )

    # ========================================================
    # FORWARD PASS
    # ========================================================
    def forward(self, x):

        encoded = self.encoder(x)

        latent_input = self.flatten(encoded)

        latent = self.encoder_fc(latent_input)

        decoded = self.decoder_fc(latent)

        decoded = decoded.view(-1, 16, 4, 16)

        output = self.decoder(decoded)

        return output


# ============================================================
# NMSE FUNCTION
# ============================================================
def calculate_nmse(actual, predicted):

    mse = torch.sum((actual - predicted) ** 2)

    power = torch.sum(actual ** 2)

    nmse = mse / (power + 1e-9)

    return nmse


# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_model():

    X = load_dataset()

    X_tensor = torch.tensor(X)

    dataset = TensorDataset(X_tensor, X_tensor)

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0
    )

    model = ResidualCsiAutoencoder().to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-6
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    print("\nStarting training...\n")

    epochs = 50

    loss_history = []

    nmse_history = []

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        total_nmse = 0

        for batch_x, _ in loader:

            batch_x = batch_x.to(device)

            output = model(batch_x)

            loss = criterion(output, batch_x)

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
        "residual_cnn_autoencoder.pth"
    )

    print("\nModel saved.")

    # ========================================================
    # LOSS PLOT
    # ========================================================
    plt.figure(figsize=(10, 5))

    plt.plot(loss_history)

    plt.title("Residual CNN Training Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.grid(True)

    plt.savefig("residual_cnn_loss.png")

    plt.close()

    # ========================================================
    # NMSE PLOT
    # ========================================================
    plt.figure(figsize=(10, 5))

    plt.plot(nmse_history)

    plt.title("Residual CNN NMSE")

    plt.xlabel("Epoch")

    plt.ylabel("NMSE")

    plt.grid(True)

    plt.savefig("residual_cnn_nmse.png")

    plt.close()

    print("Training plots saved.")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    train_model()