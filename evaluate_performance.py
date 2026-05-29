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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    latent_dim = int(checkpoint["latent_dim"])

    model = CNNAutoencoder(latent_dim=latent_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model       : {MODEL_PATH}")
    print(f"Loaded stats       : {STATS_PATH}")
    print(f"Latent dim         : {latent_dim}")
    print(f"Best val NMSE saved: {checkpoint.get('best_val_nmse', 'N/A')}")
    print(f"Best epoch saved   : {checkpoint.get('epoch', 'N/A')}")

    return model, latent_dim


def load_split_dataset(split_name):
    stats = np.load(STATS_PATH)

    x_scale = float(stats["x_scale"])
    use_angular_domain = bool(stats["use_angular_domain"])

    if split_name == "train":
        selected_idx = stats["train_idx"]
    elif split_name == "val":
        selected_idx = stats["val_idx"]
    elif split_name == "test":
        selected_idx = stats["test_idx"]
    else:
        raise ValueError("split_name must be train, val, or test")

    data = sio.loadmat(DATASET_PATH)
    H = data[MAT_KEY]

    H = H.transpose(2, 0, 1)

    if use_angular_domain:
        H = angular_domain_transform(H)

    X = complex_to_channels(H)
    X = X / x_scale

    return X[selected_idx].astype(np.float32)


def reconstruct(model, X, batch_size=256):
    outputs = []

    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            batch = torch.tensor(X[start:start + batch_size]).to(device)
            output = model(batch)
            outputs.append(output.cpu().numpy())

    return np.concatenate(outputs, axis=0)


def calculate_metrics(X, reconstructed):
    mse_sum = np.sum((X - reconstructed) ** 2)
    power_sum = np.sum(X ** 2)

    mse = np.mean((X - reconstructed) ** 2)
    power = np.mean(X ** 2)

    nmse = mse_sum / (power_sum + 1e-12)
    nmse_db = 10.0 * np.log10(nmse + 1e-12)

    return mse, power, nmse, nmse_db


def evaluate_split(model, split_name):
    X = load_split_dataset(split_name)
    reconstructed = reconstruct(model, X)

    mse, power, nmse, nmse_db = calculate_metrics(X, reconstructed)

    print("\n============================================================")
    print(f"{split_name.upper()} SPLIT")
    print("============================================================")
    print(f"Samples          : {X.shape[0]}")
    print(f"MSE              : {mse:.6e}")
    print(f"Power            : {power:.6e}")
    print(f"NMSE             : {nmse:.6f}")
    print(f"NMSE dB          : {nmse_db:.2f} dB")
    print(f"Original mean abs: {np.mean(np.abs(X)):.6e}")
    print(f"Recon mean abs   : {np.mean(np.abs(reconstructed)):.6e}")

    return X, reconstructed, nmse, nmse_db


def visualize(X, reconstructed, latent_dim):
    original = X[0]
    recon = reconstructed[0]

    original_mag = np.sqrt(original[0] ** 2 + original[1] ** 2)
    recon_mag = np.sqrt(recon[0] ** 2 + recon[1] ** 2)
    error_mag = np.abs(original_mag - recon_mag)

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original_mag, aspect="auto")
    plt.title("Original Angular CSI")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(recon_mag, aspect="auto")
    plt.title("Reconstructed Angular CSI")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(error_mag, aspect="auto")
    plt.title("Magnitude Error")
    plt.colorbar()

    plt.tight_layout()
    output_path = f"mmwave_rescnn_rms_reconstruction_latent{latent_dim}.png"
    plt.savefig(output_path)
    plt.close()

    print(f"\nSaved: {output_path}")


def main():
    model, latent_dim = load_model()

    _, _, train_nmse, train_nmse_db = evaluate_split(model, "train")
    _, _, val_nmse, val_nmse_db = evaluate_split(model, "val")
    X_test, reconstructed_test, test_nmse, test_nmse_db = evaluate_split(model, "test")

    input_dim = 2 * 16 * 64
    compression_ratio = input_dim / latent_dim
    overhead_reduction = 100.0 * (1.0 - latent_dim / input_dim)

    print("\n============================================================")
    print("FINAL SUMMARY")
    print("============================================================")
    print(f"Latent dim         : {latent_dim}")
    print(f"Compression ratio  : {compression_ratio:.1f}x")
    print(f"Overhead reduction : {overhead_reduction:.2f}%")
    print(f"Train NMSE         : {train_nmse:.6f} ({train_nmse_db:.2f} dB)")
    print(f"Val NMSE           : {val_nmse:.6f} ({val_nmse_db:.2f} dB)")
    print(f"Test NMSE          : {test_nmse:.6f} ({test_nmse_db:.2f} dB)")

    visualize(X_test, reconstructed_test, latent_dim)


if __name__ == "__main__":
    main()