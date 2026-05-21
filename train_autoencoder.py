import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# DEVICE CONFIGURATION
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# ============================================================
# DATA LOADING AND PREPROCESSING
# ============================================================
def load_dataset():

    print("Loading dataset...")

    data = sio.loadmat("CSI_dataset.mat")
    H = data["H_dataset"]

    print("Original shape:", H.shape)

    # --------------------------------------------------------
    # TRANSPOSE
    # Original: (16, 64, 10000)
    # New:      (10000, 16, 64)
    # --------------------------------------------------------
    H = H.transpose(2, 0, 1)

    print("Transposed shape:", H.shape)

    # --------------------------------------------------------
    # SEPARATE REAL AND IMAGINARY PARTS
    # --------------------------------------------------------
    H_real = np.real(H)
    H_imag = np.imag(H)

    # --------------------------------------------------------
    # STACK AS CHANNELS
    # Shape:
    # (samples, 2, 16, 64)
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
# CONVOLUTIONAL CSI AUTOENCODER
# ============================================================
class ConvolutionalCsiAutoencoder(nn.Module):

    def __init__(self):

        super(ConvolutionalCsiAutoencoder, self).__init__()

        # ====================================================
        # ENCODER
        # ====================================================
        self.encoder_conv = nn.Sequential(

            nn.Conv2d(
                in_channels=2,
                out_channels=8,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(8),

            nn.LeakyReLU(0.3),

            nn.Conv2d(
                in_channels=8,
                out_channels=4,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(4),

            nn.LeakyReLU(0.3)
        )

        # ====================================================
        # FLATTEN + BOTTLENECK
        # ====================================================

        # 4 × 16 × 64 = 4096

        self.flatten = nn.Flatten()

        # ----------------------------------------------------
        # 512 latent neurons = 8x compression
        # ----------------------------------------------------
        self.encoder_fc = nn.Linear(4096, 512)

        # ====================================================
        # DECODER FC
        # ====================================================
        self.decoder_fc = nn.Linear(512, 4096)

        # ====================================================
        # DECODER CONV
        # ====================================================
        self.decoder_conv = nn.Sequential(

            nn.ConvTranspose2d(
                in_channels=4,
                out_channels=8,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(8),

            nn.LeakyReLU(0.3),

            nn.ConvTranspose2d(
                in_channels=8,
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

        # ----------------------------------------------------
        # ENCODER
        # ----------------------------------------------------
        x = self.encoder_conv(x)

        x = self.flatten(x)

        latent = self.encoder_fc(x)

        # ----------------------------------------------------
        # DECODER
        # ----------------------------------------------------
        x = self.decoder_fc(latent)

        x = x.view(-1, 4, 16, 64)

        x = self.decoder_conv(x)

        return x


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

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------
    X = load_dataset()

    X_tensor = torch.tensor(X)

    dataset = TensorDataset(X_tensor, X_tensor)

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------
    model = ConvolutionalCsiAutoencoder().to(device)

    # --------------------------------------------------------
    # LOSS FUNCTION
    # --------------------------------------------------------
    criterion = nn.MSELoss()

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-6
    )

    # --------------------------------------------------------
    # LEARNING RATE SCHEDULER
    # --------------------------------------------------------
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10
    )

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------
    print("\nStarting training...\n")

    epochs = 200

    loss_history = []

    nmse_history = []

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        total_nmse = 0

        for batch_x, _ in loader:

            batch_x = batch_x.to(device)

            # ------------------------------------------------
            # FORWARD
            # ------------------------------------------------
            output = model(batch_x)

            # ------------------------------------------------
            # LOSS
            # ------------------------------------------------
            loss = criterion(output, batch_x)

            nmse = calculate_nmse(batch_x, output)

            # ------------------------------------------------
            # BACKPROP
            # ------------------------------------------------
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            total_nmse += nmse.item()

        # ----------------------------------------------------
        # AVERAGES
        # ----------------------------------------------------
        avg_loss = total_loss / len(loader)

        avg_nmse = total_nmse / len(loader)

        loss_history.append(avg_loss)

        nmse_history.append(avg_nmse)

        # ----------------------------------------------------
        # UPDATE LR
        # ----------------------------------------------------
        scheduler.step(avg_loss)

        # ----------------------------------------------------
        # PRINT METRICS
        # ----------------------------------------------------
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
        "cnn_autoencoder_model.pth"
    )

    print("\nTraining complete.")
    print("Model saved as cnn_autoencoder_model.pth")

    # ========================================================
    # PLOT TRAINING LOSS
    # ========================================================
    plt.figure(figsize=(10, 5))

    plt.plot(loss_history)

    plt.title("CNN Autoencoder Training Loss")

    plt.xlabel("Epoch")

    plt.ylabel("MSE Loss")

    plt.grid(True)

    plt.savefig("cnn_training_loss.png")

    plt.show()

    # ========================================================
    # PLOT NMSE
    # ========================================================
    plt.figure(figsize=(10, 5))

    plt.plot(nmse_history)

    plt.title("CNN Autoencoder NMSE")

    plt.xlabel("Epoch")

    plt.ylabel("NMSE")

    plt.grid(True)

    plt.savefig("cnn_nmse_curve.png")

    plt.show()

    print("Training plots saved.")


# ============================================================
# MAIN ENTRY
# ============================================================
if __name__ == "__main__":

    train_model()