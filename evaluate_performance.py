import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import torch

from train_autoencoder import (
    CNNAutoencoder,
    DATASET_PATH,
    MAT_KEY,
    MODEL_PATH,
    STATS_PATH,
    angular_domain_transform,
    complex_to_channels,
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    latent_dim = int(checkpoint["latent_dim"])

    model = CNNAutoencoder(latent_dim=latent_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model: {MODEL_PATH}")
    print(f"Latent dim  : {latent_dim}")

    return model, latent_dim


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_dataset():
    stats = np.load(STATS_PATH)

    x_max = float(stats["x_max"])
    test_idx = stats["test_idx"]

    use_angular_domain = bool(stats["use_angular_domain"])
    debug_tiny_overfit = bool(stats["debug_tiny_overfit"])

    data = sio.loadmat(DATASET_PATH)
    H = data[MAT_KEY]

    print("Original dataset shape:", H.shape)

    H = H.transpose(2, 0, 1)

    if use_angular_domain:
        print("Evaluation using angular-domain CSI.")
        H_processed = angular_domain_transform(H)
    else:
        print("Evaluation using raw CSI.")
        H_processed = H

    X = complex_to_channels(H_processed)

    X = X / x_max

    if debug_tiny_overfit:
        print("DEBUG_TINY_OVERFIT was enabled during training.")
        # For tiny overfit debugging, evaluate the same samples used for training.
        train_idx = stats["train_idx"]
        X_test = X[train_idx]
    else:
        X_test = X[test_idx]

    print(f"Loaded stats: {STATS_PATH}")
    print("Evaluation input shape:", X_test.shape)

    return X_test.astype(np.float32)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, latent_dim):
    print("\nEvaluating CSI reconstruction performance")
    print("Using device:", device)

    X_test = load_test_dataset()

    test_tensor = torch.tensor(X_test).to(device)

    with torch.no_grad():
        reconstructed = model(test_tensor)

    reconstructed = reconstructed.cpu().numpy()

    mse_sum = np.sum((X_test - reconstructed) ** 2)
    power_sum = np.sum(X_test ** 2)

    mse = np.mean((X_test - reconstructed) ** 2)
    power = np.mean(X_test ** 2)

    nmse = mse_sum / (power_sum + 1e-12)
    nmse_db = 10.0 * np.log10(nmse + 1e-12)

    input_dim = 2 * 16 * 64
    compression_ratio = input_dim / latent_dim

    print("\n============================================================")
    print("FINAL EVALUATION SUMMARY")
    print("============================================================")
    print(f"Samples            : {X_test.shape[0]}")
    print(f"MSE                : {mse:.6e}")
    print(f"Power              : {power:.6e}")
    print(f"NMSE               : {nmse:.6f}")
    print(f"NMSE dB            : {nmse_db:.2f} dB")
    print(f"Latent dim         : {latent_dim}")
    print(f"Compression ratio  : {compression_ratio:.1f}x")

    print("\n============================================================")
    print("SANITY CHECKS")
    print("============================================================")
    print(f"Original mean abs  : {np.mean(np.abs(X_test)):.6e}")
    print(f"Recon mean abs     : {np.mean(np.abs(reconstructed)):.6e}")
    print(f"Original max abs   : {np.max(np.abs(X_test)):.6e}")
    print(f"Recon max abs      : {np.max(np.abs(reconstructed)):.6e}")

    return X_test, reconstructed


# ============================================================
# VISUALIZATION
# ============================================================

def visualize(X_test, reconstructed, latent_dim, sample_index=0):
    print("\nVisualizing CSI reconstruction")

    sample_original = X_test[sample_index]
    sample_reconstructed = reconstructed[sample_index]

    original_mag = np.sqrt(
        sample_original[0] ** 2 +
        sample_original[1] ** 2
    )

    reconstructed_mag = np.sqrt(
        sample_reconstructed[0] ** 2 +
        sample_reconstructed[1] ** 2
    )

    error_mag = np.abs(original_mag - reconstructed_mag)

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original_mag, aspect="auto")
    plt.title("Original CSI Magnitude")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(reconstructed_mag, aspect="auto")
    plt.title("Reconstructed CSI Magnitude")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(error_mag, aspect="auto")
    plt.title("Magnitude Error")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(f"debug_csi_reconstruction_latent{latent_dim}.png")
    plt.close()

    print(f"Saved: debug_csi_reconstruction_latent{latent_dim}.png")


# ============================================================
# MAIN
# ============================================================

def main():
    model, latent_dim = load_model()

    X_test, reconstructed = evaluate(model, latent_dim)

    visualize(X_test, reconstructed, latent_dim, sample_index=0)


if __name__ == "__main__":
    main()