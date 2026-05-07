import numpy as np
import torch
import scipy.io as sio
import matplotlib.pyplot as plt

from train_autoencoder import Autoencoder


# ============================================================
# LOAD DATASET (MUST MATCH TRAINING)
# ============================================================
def load_dataset():

    print("Loading dataset...")

    data = sio.loadmat("CSI_dataset.mat")
    H = data["H_dataset"]

    print("Original shape:", H.shape)

    # reshape to (samples, antennas, subcarriers)
    H = H.transpose(2,0,1)

    print("Transposed shape:", H.shape)

    # split real and imaginary
    H_real = np.real(H)
    H_imag = np.imag(H)

    H = np.concatenate([H_real, H_imag], axis=2)

    # flatten
    X = H.reshape(H.shape[0], -1)

    print("Flattened shape:", X.shape)

    # SAME normalization used in training
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    return X.astype(np.float32)


# ============================================================
# NMSE FUNCTION
# ============================================================
def compute_nmse(original, reconstructed):

    numerator = np.sum((original - reconstructed) ** 2, axis=1)

    denominator = np.sum(original ** 2, axis=1) + 1e-8

    nmse = np.mean(numerator / denominator)

    return nmse


# ============================================================
# EVALUATION
# ============================================================
def evaluate():

    print("============================================================")
    print("Evaluating CSI Compression Performance")
    print("============================================================")

    X = load_dataset()

    model = Autoencoder()

    model.load_state_dict(
        torch.load("autoencoder_model.pth", map_location="cpu")
    )

    model.eval()

    X_tensor = torch.tensor(X)

    with torch.no_grad():

        reconstructed = model(X_tensor).numpy()

    mse = np.mean((X - reconstructed) ** 2)

    nmse = compute_nmse(X, reconstructed)

    compression_ratio = 2048 / 128

    print("\n============================================================")
    print("FINAL SUMMARY")
    print("============================================================")

    print(f"MSE  : {mse:.6f}")
    print(f"NMSE : {nmse:.6f}")
    print(f"CR   : {compression_ratio:.1f}x")

    return X, reconstructed


# ============================================================
# VISUALIZATION
# ============================================================
def visualize(X, reconstructed):

    print("\nVisualizing CSI Reconstruction")

    sample = 0

    # split real/imag
    real_orig = X[sample][:1024].reshape(16,64)
    imag_orig = X[sample][1024:].reshape(16,64)

    real_rec = reconstructed[sample][:1024].reshape(16,64)
    imag_rec = reconstructed[sample][1024:].reshape(16,64)

    # magnitude
    original_mag = np.sqrt(real_orig**2 + imag_orig**2)
    reconstructed_mag = np.sqrt(real_rec**2 + imag_rec**2)

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.title("Original CSI Magnitude")
    plt.imshow(original_mag, aspect='auto')
    plt.colorbar()

    plt.subplot(1,2,2)
    plt.title("Reconstructed CSI Magnitude")
    plt.imshow(reconstructed_mag, aspect='auto')
    plt.colorbar()

    plt.tight_layout()

    plt.savefig("csi_reconstruction.png")

    plt.show()

    print("Saved: csi_reconstruction.png")


# ============================================================
# MAIN
# ============================================================
def main():

    X, reconstructed = evaluate()

    visualize(X, reconstructed)


if __name__ == "__main__":

    main()