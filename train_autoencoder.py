import random
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


DATASET_PATH = "CSI_dataset_mmwave.mat"
MAT_KEY = "H_dataset"

SEED = 42
BATCH_SIZE = 128
EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-7
GRAD_CLIP_NORM = 1.0

LATENT_DIM = 256

MODEL_PATH = f"mmwave_rescnn_rms_latent{LATENT_DIM}.pth"
STATS_PATH = f"mmwave_rescnn_rms_stats_latent{LATENT_DIM}.npz"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
USE_ANGULAR_DOMAIN = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_raw_csi():
    data = sio.loadmat(DATASET_PATH)
    H = data[MAT_KEY]
    print("Original shape:", H.shape)
    H = H.transpose(2, 0, 1)
    print("Transposed shape:", H.shape)
    return H


def angular_domain_transform(H):
    return np.fft.fft2(H, axes=(1, 2), norm="ortho")


def complex_to_channels(H_complex):
    return np.stack(
        [np.real(H_complex), np.imag(H_complex)],
        axis=1,
    ).astype(np.float32)


def prepare_datasets():
    H = load_raw_csi()

    if USE_ANGULAR_DOMAIN:
        print("Using angular-domain CSI.")
        H = angular_domain_transform(H)
    else:
        print("Using raw CSI.")

    X = complex_to_channels(H)
    print("Input tensor shape:", X.shape)

    num_samples = X.shape[0]
    indices = np.random.permutation(num_samples)

    train_end = int(TRAIN_RATIO * num_samples)
    val_end = int((TRAIN_RATIO + VAL_RATIO) * num_samples)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]

    # RMS normalization prevents sparse angular-domain values from collapsing.
    x_scale = np.sqrt(np.mean(X_train ** 2)) + 1e-12

    X_train = X_train / x_scale
    X_val = X_val / x_scale
    X_test = X_test / x_scale

    np.savez(
        STATS_PATH,
        x_scale=x_scale,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        latent_dim=LATENT_DIM,
        use_angular_domain=USE_ANGULAR_DOMAIN,
    )

    print("RMS normalization complete.")
    print(f"RMS scale     : {x_scale:.6e}")
    print(f"Train samples : {len(X_train)}")
    print(f"Val samples   : {len(X_val)}")
    print(f"Test samples  : {len(X_test)}")

    return X_train.astype(np.float32), X_val.astype(np.float32)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.activation(x + self.block(x))


class CNNAutoencoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(16),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(32),

            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(16),

            nn.Conv2d(16, 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.flatten = nn.Flatten()
        self.encoder_fc = nn.Linear(4 * 16 * 64, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, 4 * 16 * 64)

        self.decoder_conv = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(16),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(32),

            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(16),

            nn.Conv2d(16, 2, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.encoder_conv(x)
        x = self.flatten(x)
        latent = self.encoder_fc(x)
        x = self.decoder_fc(latent)
        x = x.view(-1, 4, 16, 64)
        return self.decoder_conv(x)


def nmse_from_sums(mse_sum, power_sum):
    nmse = mse_sum / (power_sum + 1e-12)
    nmse_db = 10.0 * np.log10(nmse + 1e-12)
    return nmse, nmse_db


def evaluate_epoch(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_mse_sum = 0.0
    total_power_sum = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)

            output = model(batch_x)
            loss = criterion(output, batch_x)

            batch_size = batch_x.size(0)

            total_loss += loss.item() * batch_size
            total_mse_sum += torch.sum((batch_x - output) ** 2).item()
            total_power_sum += torch.sum(batch_x ** 2).item()
            total_samples += batch_size

    loss = total_loss / total_samples
    nmse, nmse_db = nmse_from_sums(total_mse_sum, total_power_sum)

    return loss, nmse, nmse_db


def train_model():
    set_seed(SEED)
    print("Using device:", device)

    X_train, X_val = prepare_datasets()

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(X_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(X_val)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = CNNAutoencoder(latent_dim=LATENT_DIM).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
    )

    best_val_nmse = float("inf")

    train_nmse_history = []
    val_nmse_history = []

    print("\nStarting training...\n")

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0.0
        total_mse_sum = 0.0
        total_power_sum = 0.0
        total_samples = 0

        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)

            output = model(batch_x)
            loss = criterion(output, batch_x)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRAD_CLIP_NORM,
            )

            optimizer.step()

            batch_size = batch_x.size(0)

            total_loss += loss.item() * batch_size
            total_mse_sum += torch.sum((batch_x - output) ** 2).item()
            total_power_sum += torch.sum(batch_x ** 2).item()
            total_samples += batch_size

        train_loss = total_loss / total_samples
        train_nmse, train_nmse_db = nmse_from_sums(total_mse_sum, total_power_sum)

        val_loss, val_nmse, val_nmse_db = evaluate_epoch(
            model,
            val_loader,
            criterion,
        )

        scheduler.step(val_loss)

        train_nmse_history.append(train_nmse_db)
        val_nmse_history.append(val_nmse_db)

        print(
            f"Epoch {epoch + 1:03d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.6e} | "
            f"Train NMSE: {train_nmse:.6f} ({train_nmse_db:.2f} dB) | "
            f"Val Loss: {val_loss:.6e} | "
            f"Val NMSE: {val_nmse:.6f} ({val_nmse_db:.2f} dB)"
        )

        if val_nmse < best_val_nmse:
            best_val_nmse = val_nmse

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "latent_dim": LATENT_DIM,
                    "best_val_nmse": best_val_nmse,
                    "epoch": epoch + 1,
                    "use_angular_domain": USE_ANGULAR_DOMAIN,
                },
                MODEL_PATH,
            )

    print("\nTraining complete.")
    print(f"Best model saved as {MODEL_PATH}")

    plt.figure(figsize=(10, 5))
    plt.plot(train_nmse_history, label="Train")
    plt.plot(val_nmse_history, label="Validation")
    plt.title("mmWave ResCNN NMSE")
    plt.xlabel("Epoch")
    plt.ylabel("NMSE dB")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"mmwave_rescnn_rms_nmse_latent{LATENT_DIM}.png")
    plt.close()


if __name__ == "__main__":
    train_model()