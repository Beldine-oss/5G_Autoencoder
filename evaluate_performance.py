# ============================================================
# evaluate_performance.py
# Residual CNN Evaluation
# ============================================================

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch

from train_autoencoder import ResidualCsiAutoencoder

# ============================================================
# DEVICE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# ============================================================
# LOAD DATASET
# ============================================================
def load_dataset():

    print("\nLoading dataset...")

    data = sio.loadmat("CSI_dataset.mat")

    H = data["H_dataset"]

    print("Original shape:", H.shape)

    H = H.transpose(2,0,1)

    print("Transposed shape:", H.shape)

    H_real = np.real(H)

    H_imag = np.imag(H)

    X = np.stack([H_real, H_imag], axis=1)

    print("CNN Input shape:", X.shape)

    X_max = np.max(np.abs(X))

    X = X / X_max

    print("Normalization complete.")

    return X.astype(np.float32)


# ============================================================
# LOAD MODEL
# ============================================================
def load_model():

    model = ResidualCsiAutoencoder().to(device)

    model.load_state_dict(
        torch.load(
            "residual_cnn_autoencoder.pth",
            map_location=device
        )
    )

    model.eval()

    return model


# ============================================================
# EVALUATION
# ============================================================
def evaluate(model):

    X = load_dataset()

    X_tensor = torch.tensor(X).to(device)

    with torch.no_grad():

        reconstructed = model(X_tensor)

    reconstructed = reconstructed.cpu().numpy()

    # --------------------------------------------------------
    # FLATTEN FOR METRICS
    # --------------------------------------------------------
    actual = X.reshape(X.shape[0], -1)

    predicted = reconstructed.reshape(
        reconstructed.shape[0],
        -1
    )

    mse = np.mean((actual - predicted) ** 2)

    power = np.mean(actual ** 2)

    nmse = mse / (power + 1e-9)

    compression_ratio = 2048 / 512

    print("\n=================================================")
    print("FINAL SUMMARY")
    print("=================================================")

    print(f"MSE  : {mse:.6f}")

    print(f"NMSE : {nmse:.6f}")

    print(f"CR   : {compression_ratio:.1f}x")

    return X, reconstructed


# ============================================================
# VISUALIZATION
# ============================================================
def visualize(original, reconstructed):

    sample_original = original[0]

    sample_reconstructed = reconstructed[0]

    # --------------------------------------------------------
    # MAGNITUDE
    # --------------------------------------------------------
    original_mag = np.sqrt(
        sample_original[0]**2 +
        sample_original[1]**2
    )

    reconstructed_mag = np.sqrt(
        sample_reconstructed[0]**2 +
        sample_reconstructed[1]**2
    )

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)

    plt.imshow(original_mag, aspect='auto')

    plt.title("Original CSI Magnitude")

    plt.colorbar()

    plt.subplot(1,2,2)

    plt.imshow(reconstructed_mag, aspect='auto')

    plt.title("Reconstructed CSI Magnitude")

    plt.colorbar()

    plt.tight_layout()

    plt.savefig("residual_cnn_reconstruction.png")

    plt.show()

    print("\nSaved: residual_cnn_reconstruction.png")


# ============================================================
# MAIN
# ============================================================
def main():

    print("=================================================")
    print("Residual CNN CSI Compression Evaluation")
    print("=================================================")

    model = load_model()

    original, reconstructed = evaluate(model)

    visualize(original, reconstructed)


# ============================================================
# ENTRY
# ============================================================
if __name__ == "__main__":

    main()