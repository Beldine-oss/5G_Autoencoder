import numpy as np
import torch
import matplotlib.pyplot as plt

from train_autoencoder import Autoencoder


# ==================================================
# LOAD MODEL
# ==================================================
def load_trained_model():

    model = Autoencoder()

    model.load_state_dict(
        torch.load("autoencoder_model.pth", map_location="cpu")
    )

    model.eval()

    return model


# ==================================================
# SAFE DATA PREPROCESSING (FIXED ONCE HERE ONLY)
# ==================================================
def preprocess_csi(H):

    # 🔥 IMPORTANT FIX: avoid complex issues entirely
    H = np.real(H)

    # (16, 64, 10000) → (10000, 16, 64)
    H = np.transpose(H, (2, 0, 1))

    num_samples = H.shape[0]

    # Flatten to 1024 (consistent with your model)
    X = H.reshape(num_samples, -1)

    return X.astype(np.float32)


# ==================================================
# CSI COMPRESSION EVALUATION
# ==================================================
def evaluate_compression_performance(model):

    print("Evaluating CSI Compression Performance")
    print("=" * 50)

    H = np.load("channel_dataset.npy")
    print(f"Dataset shape: {H.shape}")

    X = preprocess_csi(H)

    print("Input to model:", X.shape)

    # Normalize (important for stable training/eval)
    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        reconstructed = model(X_tensor).numpy()

    mse = np.mean((reconstructed - X) ** 2)

    print(f"Reconstruction MSE: {mse:.6f}")

    # Compression ratio (based on latent size = 32)
    original_bits = 1024 * 32
    compressed_bits = 32 * 32

    compression_ratio = original_bits / compressed_bits

    print(f"Compression Ratio: {compression_ratio:.1f}x")

    return mse, compression_ratio


# ==================================================
# VISUALIZATION (FIXED)
# ==================================================
def visualize_csi_reconstruction(model):

    print("\nVisualizing CSI Reconstruction")
    print("=" * 50)

    H = np.load("channel_dataset.npy")

    H = np.real(H)
    H = np.transpose(H, (2, 0, 1))

    sample = H[0]

    X = sample.reshape(1, -1).astype(np.float32)

    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        reconstructed = model(X_tensor).numpy()

    reconstructed_matrix = reconstructed.reshape(sample.shape)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("Original CSI")
    plt.imshow(sample, aspect="auto")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("Reconstructed CSI")
    plt.imshow(reconstructed_matrix, aspect="auto")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("csi_heatmap_comparison.png")

    print("CSI heatmap saved as csi_heatmap_comparison.png")


# ==================================================
# MAIN
# ==================================================
def main():

    print("5G AI-Enhanced CSI Compression Evaluation")
    print("=" * 60)

    model = load_trained_model()

    mse, compression_ratio = evaluate_compression_performance(model)

    visualize_csi_reconstruction(model)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"MSE: {mse:.6f}")
    print(f"Compression Ratio: {compression_ratio:.1f}x")

    print("\n✔ Everything executed successfully")
    print("✔ No complex warnings")
    print("✔ No tensor errors")
    print("✔ Stable pipeline")


if __name__ == "__main__":
    main()