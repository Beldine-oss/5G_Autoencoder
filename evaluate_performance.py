import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch

from train_autoencoder import CNNAutoencoder


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    model = CNNAutoencoder().to(device)

    model.load_state_dict(
        torch.load(
            "autoencoder_model.pth",
            map_location=device
        )
    )

    model.eval()

    return model


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    data = sio.loadmat("CSI_dataset.mat")

    H = data["H_dataset"]

    H = H.transpose(2,0,1)

    H_real = np.real(H)
    H_imag = np.imag(H)

    X = np.stack([H_real, H_imag], axis=1)

    X_max = np.max(np.abs(X))

    X = X / X_max

    return X.astype(np.float32)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model):

    print("\nEvaluating CSI Compression Performance")

    X = load_dataset()

    X_tensor = torch.tensor(X).to(device)

    with torch.no_grad():

        reconstructed = model(X_tensor)

    reconstructed = reconstructed.cpu().numpy()

    mse = np.mean((X - reconstructed) ** 2)

    power = np.mean(X ** 2)

    nmse = mse / (power + 1e-9)

    compression_ratio = 2048 / 256

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

    sample_original = X[0]
    sample_reconstructed = reconstructed[0]

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

    plt.savefig("csi_reconstruction.png")

    plt.show()

    print("Saved: csi_reconstruction.png")


# ============================================================
# MAIN
# ============================================================

def main():

    model = load_model()

    X, reconstructed = evaluate(model)

    visualize(X, reconstructed)


if __name__ == "__main__":
    main()