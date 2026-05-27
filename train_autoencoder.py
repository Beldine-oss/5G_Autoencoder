import random
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = "CSI_dataset.mat"
MAT_KEY = "H_dataset"

SEED = 42
BATCH_SIZE = 128
EPOCHS = 300
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
GRAD_CLIP_NORM = 1.0

LATENT_DIM = 2048

MODEL_PATH = f"debug_autoencoder_model_latent{LATENT_DIM}.pth"
STATS_PATH = f"debug_preprocessing_stats_latent{LATENT_DIM}.npz"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

DEBUG_TINY_OVERFIT = True
DEBUG_NUM_SAMPLES = 200

USE_ANGULAR_DOMAIN = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATA PREPROCESSING
# ============================================================

def load_raw_csi():
    print("Loading dataset...")

    data = sio.loadmat(DATASET_PATH)
    H = data[MAT_KEY]

    print("Original shape:", H.shape)

    H = H.transpose(2, 0, 1)

    print("Transposed shape:", H.shape)

    return H


def angular_domain_transform(H):
    return np.fft.fft2(H, axes=(1, 2), norm="ortho")


def complex_to_channels(H_complex):
    H_real = np.real(H_complex)
    H_imag = np.imag(H_complex)

    X = np.stack([H_real, H_imag], axis=1)

    return X.astype(np.float32)


def prepare_datasets():
    H = load_raw_csi()

    if USE_ANGULAR_DOMAIN:
        print("Using angular-domain CSI.")
        H_processed = angular_domain_transform(H)
    else:
        print("Using raw CSI.")
        H_processed = H

    X = complex_to_channels(H_processed)

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

    x_max = np.max(np.abs(X_train)) + 1e-12

    X_train = X_train / x_max
    X_val = X_val / x_max
    X_test = X_test / x_max

    if DEBUG_TINY_OVERFIT:
        print("DEBUG_TINY_OVERFIT is enabled.")
        X_train = X_train[:DEBUG_NUM_SAMPLES]
        X_val = X_train.copy()

        train_idx = train_idx[:DEBUG_NUM_SAMPLES]
        val_idx = train_idx.copy()

    np.savez(
        STATS_PATH,
        x_max=x_max,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        latent_dim=LATENT_DIM,
        use_angular_domain=USE_ANGULAR_DOMAIN,
        debug_tiny_overfit=DEBUG_TINY_OVERFIT,
    )

    print("Normalization complete.")
    print(f"Train samples: {len(X_train)}")
    print(f"Val samples  : {len(X_val)}")
    print(f"Test samples : {len(X_test)}")
    print(f"Saved preprocessing stats to {STATS_PATH}")

    return X_train.astype(np.float32), X_val.astype(np.float32)


# ============================================================
# MODEL
# ============================================================

class CNNAutoencoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()

        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 2048),
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        reconstruction = reconstruction.view(-1, 2, 16, 64)

        return reconstruction


# ============================================================
# METRICS
# ============================================================

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
            batch_x = batch_x.to(device, non_blocking=True)

            output = model(batch_x)
            loss = criterion(output, batch_x)

            batch_size = batch_x.size(0)

            total_loss += loss.item() * batch_size
            total_mse_sum += torch.sum((batch_x - output) ** 2).item()
            total_power_sum += torch.sum(batch_x ** 2).item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    nmse, nmse_db = nmse_from_sums(total_mse_sum, total_power_sum)

    return avg_loss, nmse, nmse_db


# ============================================================
# TRAINING
# ============================================================

def train_model():
    set_seed(SEED)

    print("Using device:", device)

    X_train, X_val = prepare_datasets()

    train_tensor = torch.tensor(X_train)
    val_tensor = torch.tensor(X_val)

    num_workers = 2 if torch.cuda.is_available() else 0

    train_loader = DataLoader(
        TensorDataset(train_tensor, train_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        TensorDataset(val_tensor, val_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = CNNAutoencoder(latent_dim=LATENT_DIM).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=30,
    )

    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_nmse = float("inf")

    train_loss_history = []
    val_loss_history = []
    train_nmse_db_history = []
    val_nmse_db_history = []

    print("\nStarting training...\n")

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0.0
        total_mse_sum = 0.0
        total_power_sum = 0.0
        total_samples = 0

        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(batch_x)
                loss = criterion(output, batch_x)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRAD_CLIP_NORM,
            )

            scaler.step(optimizer)
            scaler.update()

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

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_nmse_db_history.append(train_nmse_db)
        val_nmse_db_history.append(val_nmse_db)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1:03d}/{EPOCHS} | "
            f"LR: {current_lr:.2e} | "
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
                    "debug_tiny_overfit": DEBUG_TINY_OVERFIT,
                },
                MODEL_PATH,
            )

    print("\nTraining complete.")
    print(f"Best model saved as {MODEL_PATH}")

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label="Train")
    plt.plot(val_loss_history, label="Validation")
    plt.title("Debug Autoencoder Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"debug_training_loss_latent{LATENT_DIM}.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(train_nmse_db_history, label="Train")
    plt.plot(val_nmse_db_history, label="Validation")
    plt.title("Debug Autoencoder NMSE")
    plt.xlabel("Epoch")
    plt.ylabel("NMSE dB")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"debug_nmse_curve_latent{LATENT_DIM}.png")
    plt.close()

    print("Training plots saved.")


if __name__ == "__main__":
    train_model()