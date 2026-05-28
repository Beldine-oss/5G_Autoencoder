import numpy as np
import scipy.io as sio
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

    x_max = float(stats["x_max"])
    use_angular_domain = bool(stats["use_angular_domain"])

    if split_name == "train":
        selected_idx = stats["train_idx"]
    elif split_name == "val":
        selected_idx = stats["val_idx"]
    elif split_name == "test":
        selected_idx = stats["test_idx"]
    else:
        raise ValueError("split_name must be 'train', 'val', or 'test'")

    data = sio.loadmat(DATASET_PATH)
    H = data[MAT_KEY]

    # Original: (16, 64, samples)
    # New:      (samples, 16, 64)
    H = H.transpose(2, 0, 1)

    if use_angular_domain:
        H_processed = angular_domain_transform(H)
    else:
        H_processed = H

    X = complex_to_channels(H_processed)
    X = X / x_max

    X_split = X[selected_idx]

    return X_split.astype(np.float32)


def evaluate_split(model, split_name):
    X = load_split_dataset(split_name)

    x_tensor = torch.tensor(X).to(device)

    with torch.no_grad():
        reconstructed = model(x_tensor)

    reconstructed = reconstructed.cpu().numpy()

    mse_sum = np.sum((X - reconstructed) ** 2)
    power_sum = np.sum(X ** 2)

    mse = np.mean((X - reconstructed) ** 2)
    power = np.mean(X ** 2)

    nmse = mse_sum / (power_sum + 1e-12)
    nmse_db = 10.0 * np.log10(nmse + 1e-12)

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

    return nmse, nmse_db


def main():
    model, latent_dim = load_model()

    print("\nEvaluating saved model on all splits...")

    train_nmse, train_nmse_db = evaluate_split(model, "train")
    val_nmse, val_nmse_db = evaluate_split(model, "val")
    test_nmse, test_nmse_db = evaluate_split(model, "test")

    print("\n============================================================")
    print("SUMMARY")
    print("============================================================")
    print(f"Train NMSE: {train_nmse:.6f} ({train_nmse_db:.2f} dB)")
    print(f"Val NMSE  : {val_nmse:.6f} ({val_nmse_db:.2f} dB)")
    print(f"Test NMSE : {test_nmse:.6f} ({test_nmse_db:.2f} dB)")


if __name__ == "__main__":
    main()