import os
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


SNR_DB_VALUES = [-10, -5, 0, 5, 10, 15, 20]
NUM_STREAMS = 1
BATCH_SIZE = 256


def channels_to_complex(X):
    return X[:, 0, :, :] + 1j * X[:, 1, :, :]


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    latent_dim = int(checkpoint["latent_dim"])

    model = CNNAutoencoder(latent_dim=latent_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model : {MODEL_PATH}")
    print(f"Loaded stats : {STATS_PATH}")
    print(f"Latent dim   : {latent_dim}")

    return model, latent_dim


def reconstruct_batches(model, X):
    outputs = []

    with torch.no_grad():
        for start in range(0, X.shape[0], BATCH_SIZE):
            end = start + BATCH_SIZE
            batch = torch.tensor(X[start:end]).to(device)

            output = model(batch)

            outputs.append(output.cpu().numpy())

    return np.concatenate(outputs, axis=0)


def load_true_and_reconstructed_csi(model):
    stats = np.load(STATS_PATH)

    x_scale = float(stats["x_scale"])
    test_idx = stats["test_idx"]
    use_angular_domain = bool(stats["use_angular_domain"])

    data = sio.loadmat(DATASET_PATH)
    H = data[MAT_KEY]

    # Original: (16, 64, samples)
    # New:      (samples, 16, 64)
    H = H.transpose(2, 0, 1)

    H_true = H[test_idx]

    if use_angular_domain:
        H_model_domain = angular_domain_transform(H)
    else:
        H_model_domain = H

    X = complex_to_channels(H_model_domain)
    X = X / x_scale
    X_test = X[test_idx].astype(np.float32)

    X_reconstructed = reconstruct_batches(model, X_test)

    # De-normalize reconstructed CSI.
    X_reconstructed = X_reconstructed * x_scale

    H_reconstructed_model_domain = channels_to_complex(X_reconstructed)

    if use_angular_domain:
        H_reconstructed = np.fft.ifft2(
            H_reconstructed_model_domain,
            axes=(1, 2),
            norm="ortho",
        )
    else:
        H_reconstructed = H_reconstructed_model_domain

    return H_true, H_reconstructed


def spectral_efficiency(H_true, H_est, snr_db, num_streams=1):
    """
    Designs SVD beamformer from estimated CSI, then evaluates rate on true CSI.
    H_true shape: (Nr, Nt)
    H_est shape : (Nr, Nt)
    """
    snr_linear = 10 ** (snr_db / 10)

    U, _, Vh = np.linalg.svd(H_est, full_matrices=False)

    F = Vh.conj().T[:, :num_streams]
    W = U[:, :num_streams]

    H_eff = W.conj().T @ H_true @ F

    eye = np.eye(num_streams, dtype=np.complex128)

    rate_matrix = eye + (snr_linear / num_streams) * (H_eff @ H_eff.conj().T)

    sign, logdet = np.linalg.slogdet(rate_matrix)

    if sign <= 0:
        return 0.0

    return float(np.real(logdet / np.log(2)))


def evaluate_spectral_efficiency(H_true_all, H_hat_all, snr_values, num_streams=1):
    perfect_avg_rates = []
    ai_avg_rates = []

    for snr_db in snr_values:
        perfect_rates = []
        ai_rates = []

        for H_true, H_hat in zip(H_true_all, H_hat_all):
            perfect_rate = spectral_efficiency(
                H_true=H_true,
                H_est=H_true,
                snr_db=snr_db,
                num_streams=num_streams,
            )

            ai_rate = spectral_efficiency(
                H_true=H_true,
                H_est=H_hat,
                snr_db=snr_db,
                num_streams=num_streams,
            )

            perfect_rates.append(perfect_rate)
            ai_rates.append(ai_rate)

        perfect_avg = np.mean(perfect_rates)
        ai_avg = np.mean(ai_rates)

        perfect_avg_rates.append(perfect_avg)
        ai_avg_rates.append(ai_avg)

        loss_percent = 100.0 * (perfect_avg - ai_avg) / (perfect_avg + 1e-12)

        print(
            f"SNR {snr_db:>3} dB | "
            f"Perfect CSI: {perfect_avg:.4f} bps/Hz | "
            f"AI CSI: {ai_avg:.4f} bps/Hz | "
            f"Loss: {loss_percent:.2f}%"
        )

    return np.array(perfect_avg_rates), np.array(ai_avg_rates)


def plot_results(snr_values, perfect_rates, ai_rates, latent_dim):
    os.makedirs("images", exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(snr_values, perfect_rates, marker="o", label="Perfect CSI")
    plt.plot(snr_values, ai_rates, marker="s", label="AI-Reconstructed CSI")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Spectral Efficiency (bps/Hz)")
    plt.title(f"Spectral Efficiency Comparison, Latent {latent_dim}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_path = f"images/spectral_efficiency_latent{latent_dim}.png"
    plt.savefig(output_path)
    plt.close()

    print(f"\nSaved: {output_path}")


def main():
    model, latent_dim = load_model()

    print("\nLoading true and reconstructed CSI...")
    H_true, H_hat = load_true_and_reconstructed_csi(model)

    print("True CSI shape         :", H_true.shape)
    print("Reconstructed CSI shape:", H_hat.shape)

    print("\nEvaluating spectral efficiency...\n")

    perfect_rates, ai_rates = evaluate_spectral_efficiency(
        H_true_all=H_true,
        H_hat_all=H_hat,
        snr_values=SNR_DB_VALUES,
        num_streams=NUM_STREAMS,
    )

    plot_results(SNR_DB_VALUES, perfect_rates, ai_rates, latent_dim)

    print("\n============================================================")
    print("FINAL SPECTRAL EFFICIENCY SUMMARY")
    print("============================================================")
    for snr_db, perfect, ai in zip(SNR_DB_VALUES, perfect_rates, ai_rates):
        loss_percent = 100.0 * (perfect - ai) / (perfect + 1e-12)

        print(
            f"{snr_db:>3} dB | "
            f"Perfect: {perfect:.4f} bps/Hz | "
            f"AI: {ai:.4f} bps/Hz | "
            f"Loss: {loss_percent:.2f}%"
        )


if __name__ == "__main__":
    main()