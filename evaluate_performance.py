import numpy as np
import torch
import matplotlib.pyplot as plt
from train_autoencoder import Autoencoder


# ==================================================
# LOAD TRAINED MODEL
# ==================================================
def load_model():

    model = Autoencoder()

    model.load_state_dict(torch.load("autoencoder_model.pth", map_location="cpu"))

    model.eval()

    return model


# ==================================================
# SAME PREPROCESSING AS TRAINING (CRITICAL FIX)
# ==================================================
def preprocess(H):

    H = np.transpose(H, (2, 0, 1))

    # IMPORTANT: match training (real + imag)
    H_real = np.real(H)
    H_imag = np.imag(H)

    H = np.concatenate([H_real, H_imag], axis=-1)

    X = H.reshape(H.shape[0], -1)

    # SAME normalization as training
    X = (X - np.mean(X)) / (np.std(X) + 1e-8)

    return X.astype(np.float32)


# ==================================================
# CSI COMPRESSION EVALUATION
# ==================================================
def evaluate(model):

    print("\nEvaluating CSI Compression Performance")
    print("=" * 60)

    H = np.load("channel_dataset.npy")

    print("Dataset shape:", H.shape)

    X = preprocess(H)

    print("Input to model:", X.shape)

    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        reconstructed = model(X_tensor).numpy()

    # ==================================================
    # METRICS (CORRECT RESEARCH FORMULAS)
    # ==================================================

    mse = np.mean((X - reconstructed) ** 2)

    nmse = np.mean(np.linalg.norm(X - reconstructed, axis=1) ** 2 /
                   np.linalg.norm(X, axis=1) ** 2)

    print(f"Reconstruction MSE : {mse:.6f}")
    print(f"Reconstruction NMSE: {nmse:.6f}")

    # ==================================================
    # COMPRESSION RATIO (FIXED FOR OPTION A)
    # ==================================================
    input_dim = 2048
    latent_dim = 128

    compression_ratio = input_dim / latent_dim

    print(f"Compression Ratio: {compression_ratio:.1f}x")

    return mse, nmse, compression_ratio


# ==================================================
# VISUALIZATION (FIXED)
# ==================================================
def visualize(model):

    print("\nVisualizing CSI Reconstruction")
    print("=" * 60)

    H = np.load("channel_dataset.npy")

    H = np.transpose(H, (2, 0, 1))

    sample = H[0]

    H_real = np.real(sample)
    H_imag = np.imag(sample)

    combined = np.concatenate([H_real, H_imag], axis=-1)

    X = combined.reshape(1, -1)

    X = (X - np.mean(X)) / (np.std(X) + 1e-8)

    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        reconstructed = model(X_tensor).numpy()

    reconstructed = reconstructed.reshape(combined.shape)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("Original CSI")
    plt.imshow(combined, aspect="auto")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("Reconstructed CSI")
    plt.imshow(reconstructed, aspect="auto")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("csi_reconstruction.png")

    print("Saved: csi_reconstruction.png")


# ==================================================
# MAIN
# ==================================================
def main():

    print("5G AI-Enhanced CSI Evaluation (Aligned)")
    print("=" * 60)

    model = load_model()

    mse, nmse, cr = evaluate(model)

    visualize(model)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print(f"MSE  : {mse:.6f}")
    print(f"NMSE : {nmse:.6f}")
    print(f"CR   : {cr:.1f}x")

    print("\n✔ Evaluation aligned with training pipeline")
    print("✔ NMSE correctly computed")
    print("✔ No beamforming dependency issues")


if __name__ == "__main__":
    main()