import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io as sio

from train_autoencoder import Autoencoder


# ==================================================
# LOAD DATASET
# ==================================================
def load_dataset():

    print("Loading dataset...")

    data = sio.loadmat('CSI_dataset.mat')
    H = data['H_dataset']

    print("Original shape:", H.shape)

    # Fix dimensions
    H = H.transpose(2, 0, 1)

    print("Transposed shape:", H.shape)

    # Split real / imaginary
    H_real = np.real(H)
    H_imag = np.imag(H)

    # Create 2 channels
    H = np.stack([H_real, H_imag], axis=1)

    print("CNN input shape:", H.shape)

    # Normalize
    H = (H - np.mean(H)) / (np.std(H) + 1e-8)

    return H.astype(np.float32)


# ==================================================
# NMSE METRIC
# ==================================================
def compute_nmse(original, reconstructed):

    num = np.sum((original - reconstructed) ** 2, axis=(1,2,3))
    den = np.sum(original ** 2, axis=(1,2,3))

    nmse = np.mean(num / den)

    return nmse


# ==================================================
# LOAD TRAINED MODEL
# ==================================================
def load_model():

    model = Autoencoder()

    model.load_state_dict(
        torch.load("autoencoder_model.pth", map_location="cpu")
    )

    model.eval()

    return model


# ==================================================
# PERFORMANCE EVALUATION
# ==================================================
def evaluate_performance():

    print("5G AI-Enhanced CSI Compression Evaluation")
    print("=" * 60)

    H = load_dataset()

    model = load_model()

    X_tensor = torch.tensor(H)

    print("Input tensor shape:", X_tensor.shape)

    with torch.no_grad():

        reconstructed = model(X_tensor).numpy()

    # ==================================================
    # METRICS
    # ==================================================
    mse = np.mean((H - reconstructed) ** 2)

    nmse = compute_nmse(H, reconstructed)

    print("\nEvaluation Results")
    print("=" * 40)

    print(f"MSE  : {mse:.6f}")
    print(f"NMSE : {nmse:.6f}")

    # Compression ratio
    original_size = 16 * 64 * 2
    compressed_size = 128

    compression_ratio = original_size / compressed_size

    print(f"Compression Ratio : {compression_ratio:.1f}x")

    return H, reconstructed


# ==================================================
# VISUALIZATION
# ==================================================
def visualize_reconstruction(H, reconstructed):

    print("\nVisualizing CSI reconstruction...")

    sample = 0

    original = H[sample,0]
    recon = reconstructed[sample,0]

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.title("Original CSI (Real)")
    plt.imshow(original, aspect='auto')
    plt.colorbar()

    plt.subplot(1,2,2)
    plt.title("Reconstructed CSI (Real)")
    plt.imshow(recon, aspect='auto')
    plt.colorbar()

    plt.tight_layout()

    plt.savefig("csi_reconstruction.png")

    plt.show()

    print("Saved: csi_reconstruction.png")


# ==================================================
# MAIN
# ==================================================
def main():

    H, reconstructed = evaluate_performance()

    visualize_reconstruction(H, reconstructed)

    print("\nEvaluation complete.")


if __name__ == "__main__":

    main()