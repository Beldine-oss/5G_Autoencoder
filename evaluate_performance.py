import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ============================================================
# IMPORT CNN MODEL
# ============================================================
from train_autoencoder import ConvolutionalCsiAutoencoder


# ============================================================
# DEVICE CONFIGURATION
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# ============================================================
# LOAD AND PREPROCESS DATASET
# ============================================================
def load_dataset():

    print("\nLoading dataset...")

    data = sio.loadmat("CSI_dataset.mat")

    H = data["H_dataset"]

    print("Original shape:", H.shape)

    # --------------------------------------------------------
    # TRANSPOSE
    # (16, 64, 10000) -> (10000, 16, 64)
    # --------------------------------------------------------
    H = H.transpose(2, 0, 1)

    print("Transposed shape:", H.shape)

    # --------------------------------------------------------
    # SPLIT REAL + IMAGINARY
    # --------------------------------------------------------
    H_real = np.real(H)

    H_imag = np.imag(H)

    # --------------------------------------------------------
    # STACK CHANNELS
    # Shape:
    # (samples, 2, 16, 64)
    # --------------------------------------------------------
    X = np.stack([H_real, H_imag], axis=1)

    print("CNN Input shape:", X.shape)

    # --------------------------------------------------------
    # GLOBAL MAX NORMALIZATION
    # MUST MATCH TRAINING SCRIPT
    # --------------------------------------------------------
    X_max = np.max(np.abs(X))

    X = X / X_max

    print("Normalization complete.")

    return X.astype(np.float32)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================
def load_trained_model():

    print("\nLoading trained CNN model...")

    model = ConvolutionalCsiAutoencoder().to(device)

    model.load_state_dict(
        torch.load(
            "cnn_autoencoder_model.pth",
            map_location=device
        )
    )

    model.eval()

    print("Model loaded successfully.")

    return model


# ============================================================
# EVALUATE CSI COMPRESSION PERFORMANCE
# ============================================================
def evaluate_compression_performance(model):

    print("\n============================================================")
    print("Evaluating CSI Compression Performance")
    print("============================================================")

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------
    X = load_dataset()

    X_tensor = torch.tensor(X).to(device)

    # --------------------------------------------------------
    # RECONSTRUCTION
    # --------------------------------------------------------
    with torch.no_grad():

        reconstructed = model(X_tensor)

    reconstructed = reconstructed.cpu().numpy()

    # --------------------------------------------------------
    # FLATTEN FOR METRICS
    # --------------------------------------------------------
    actual = X.reshape(X.shape[0], -1)

    predicted = reconstructed.reshape(reconstructed.shape[0], -1)

    # --------------------------------------------------------
    # MSE
    # --------------------------------------------------------
    mse = np.mean((actual - predicted) ** 2)

    # --------------------------------------------------------
    # NMSE
    # --------------------------------------------------------
    signal_power = np.mean(actual ** 2)

    nmse = mse / (signal_power + 1e-9)

    # --------------------------------------------------------
    # COMPRESSION RATIO
    # --------------------------------------------------------
    original_size = 2048

    latent_size = 512

    compression_ratio = original_size / latent_size

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------
    print(f"\nReconstruction MSE : {mse:.6f}")

    print(f"Reconstruction NMSE: {nmse:.6f}")

    print(f"Compression Ratio: {compression_ratio:.1f}x")

    return mse, nmse, compression_ratio


# ============================================================
# VISUALIZE CSI RECONSTRUCTION
# ============================================================
def visualize_csi_reconstruction(model):

    print("\n============================================================")
    print("Visualizing CSI Reconstruction")
    print("============================================================")

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------
    X = load_dataset()

    # --------------------------------------------------------
    # SELECT SAMPLE
    # --------------------------------------------------------
    sample = X[0]

    sample_tensor = torch.tensor(
        sample[np.newaxis, ...]
    ).to(device)

    # --------------------------------------------------------
    # RECONSTRUCT
    # --------------------------------------------------------
    with torch.no_grad():

        reconstructed = model(sample_tensor)

    reconstructed = reconstructed.cpu().numpy()[0]

    # --------------------------------------------------------
    # COMPUTE MAGNITUDE
    # sqrt(real² + imag²)
    # --------------------------------------------------------
    original_mag = np.sqrt(
        sample[0] ** 2 + sample[1] ** 2
    )

    reconstructed_mag = np.sqrt(
        reconstructed[0] ** 2 + reconstructed[1] ** 2
    )

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------
    plt.figure(figsize=(12, 5))

    # --------------------------------------------------------
    # ORIGINAL CSI
    # --------------------------------------------------------
    plt.subplot(1, 2, 1)

    plt.imshow(
        original_mag,
        aspect='auto'
    )

    plt.title("Original CSI Magnitude")

    plt.xlabel("Subcarriers")

    plt.ylabel("Antennas")

    plt.colorbar()

    # --------------------------------------------------------
    # RECONSTRUCTED CSI
    # --------------------------------------------------------
    plt.subplot(1, 2, 2)

    plt.imshow(
        reconstructed_mag,
        aspect='auto'
    )

    plt.title("Reconstructed CSI Magnitude")

    plt.xlabel("Subcarriers")

    plt.ylabel("Antennas")

    plt.colorbar()

    # --------------------------------------------------------
    # SAVE FIGURE
    # --------------------------------------------------------
    plt.tight_layout()

    plt.savefig("cnn_csi_reconstruction.png")

    plt.show()

    print("\nSaved: cnn_csi_reconstruction.png")


# ============================================================
# PLOT LOSS CURVES (OPTIONAL)
# ============================================================
def plot_summary():

    print("\n============================================================")
    print("Evaluation Complete")
    print("============================================================")


# ============================================================
# MAIN FUNCTION
# ============================================================
def main():

    print("============================================================")
    print("5G AI-Enhanced CNN CSI Compression Evaluation")
    print("============================================================")

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------
    model = load_trained_model()

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------
    mse, nmse, compression_ratio = \
        evaluate_compression_performance(model)

    # --------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------
    visualize_csi_reconstruction(model)

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------
    print("\n============================================================")
    print("FINAL SUMMARY")
    print("============================================================")

    print(f"MSE  : {mse:.6f}")

    print(f"NMSE : {nmse:.6f}")

    print(f"CR   : {compression_ratio:.1f}x")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":

    main()