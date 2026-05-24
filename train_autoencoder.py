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
    # Shape:
    # (10000,2,16,64)
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
# RESIDUAL CNN AUTOENCODER
# ============================================================
class ResidualCsiAutoencoder(nn.Module):

    def __init__(self):

        super(ResidualCsiAutoencoder, self).__init__()

        # ====================================================
        # ENCODER
        # ====================================================

        self.encoder = nn.Sequential(

            nn.Conv2d(2, 16, kernel_size=3, padding=1),

            nn.BatchNorm2d(16),

            nn.LeakyReLU(0.3),

            ResidualBlock(16),

            # 16x64 -> 8x32
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),

            nn.BatchNorm2d(32),

            nn.LeakyReLU(0.3),

            ResidualBlock(32),

            # 8x32 -> 4x16
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),

            nn.BatchNorm2d(64),

            nn.LeakyReLU(0.3),

            ResidualBlock(64)
        )

        # ====================================================
        # FLATTEN
        # ====================================================

        self.flatten = nn.Flatten()

        # 64 × 4 × 16 = 4096

        # ====================================================
        # BOTTLENECK
        # ====================================================

        self.encoder_fc = nn.Linear(4096, 512)

        self.decoder_fc = nn.Linear(512, 4096)

        # ====================================================
        # DECODER
        # ====================================================

        self.decoder = nn.Sequential(

            ResidualBlock(64),

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=2,
                stride=2
            ),

            nn.BatchNorm2d(32),

            nn.LeakyReLU(0.3),

            ResidualBlock(32),

            nn.ConvTranspose2d(
                32,
                16,
                kernel_size=2,
                stride=2
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

    # ========================================================
    # FORWARD PASS
    # ========================================================
    def forward(self, x):

        encoded = self.encoder(x)

        latent_input = self.flatten(encoded)

        latent = self.encoder_fc(latent_input)

        decoded = self.decoder_fc(latent)

        decoded = decoded.view(-1, 64, 4, 16)

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
        batch_size=32,
        shuffle=True
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
        patience=10
    )

    print("\nStarting training...\n")

    epochs = 300

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

    plt.show()

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

    plt.show()

    print("Training plots saved.")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    train_model()