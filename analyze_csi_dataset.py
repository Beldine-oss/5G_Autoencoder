import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


DATASET_PATH = "CSI_dataset_mmwave.mat"
MAT_KEY = "H_dataset"

TOP_K_VALUES = [64, 128, 256, 512, 1024]


def load_csi():
    data = sio.loadmat(DATASET_PATH)
    H = data[MAT_KEY]

    print("Original shape:", H.shape)

    # Original expected: (16, 64, samples)
    # New: (samples, 16, 64)
    H = H.transpose(2, 0, 1)

    print("Transposed shape:", H.shape)
    print("Data dtype:", H.dtype)

    return H


def basic_statistics(H):
    magnitude = np.abs(H)
    power = magnitude ** 2

    print("\n============================================================")
    print("BASIC CSI STATISTICS")
    print("============================================================")
    print(f"Samples             : {H.shape[0]}")
    print(f"Rows / antennas (?) : {H.shape[1]}")
    print(f"Cols / subcarriers ?: {H.shape[2]}")
    print(f"Mean magnitude      : {np.mean(magnitude):.6e}")
    print(f"Std magnitude       : {np.std(magnitude):.6e}")
    print(f"Max magnitude       : {np.max(magnitude):.6e}")
    print(f"Mean power          : {np.mean(power):.6e}")
    print(f"Std power           : {np.std(power):.6e}")


def adjacent_correlation(H):
    """
    Measures average complex correlation between adjacent rows and columns.
    High correlation means the dataset has structure an autoencoder can exploit.
    """

    row_corrs = []
    col_corrs = []

    for sample in H:
        # Adjacent row correlation
        for i in range(sample.shape[0] - 1):
            a = sample[i, :].reshape(-1)
            b = sample[i + 1, :].reshape(-1)

            denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
            corr = np.abs(np.vdot(a, b)) / denom
            row_corrs.append(corr)

        # Adjacent column correlation
        for j in range(sample.shape[1] - 1):
            a = sample[:, j].reshape(-1)
            b = sample[:, j + 1].reshape(-1)

            denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
            corr = np.abs(np.vdot(a, b)) / denom
            col_corrs.append(corr)

    print("\n============================================================")
    print("ADJACENT CORRELATION")
    print("============================================================")
    print(f"Mean adjacent row correlation : {np.mean(row_corrs):.6f}")
    print(f"Mean adjacent col correlation : {np.mean(col_corrs):.6f}")
    print(f"Median row correlation        : {np.median(row_corrs):.6f}")
    print(f"Median col correlation        : {np.median(col_corrs):.6f}")


def angular_energy_concentration(H):
    """
    Applies 2D FFT and measures how much energy is concentrated in the
    largest coefficients.

    If top 256 coefficients contain high energy, 8x compression is plausible.
    If not, the channel is not very compressible in this representation.
    """

    H_ad = np.fft.fft2(H, axes=(1, 2), norm="ortho")

    energy = np.abs(H_ad) ** 2

    flat_energy = energy.reshape(energy.shape[0], -1)

    sorted_energy = np.sort(flat_energy, axis=1)[:, ::-1]

    total_energy = np.sum(sorted_energy, axis=1) + 1e-12

    print("\n============================================================")
    print("ANGULAR-DOMAIN ENERGY CONCENTRATION")
    print("============================================================")

    ratios = {}

    for k in TOP_K_VALUES:
        top_k_energy = np.sum(sorted_energy[:, :k], axis=1)
        ratio = top_k_energy / total_energy

        ratios[k] = ratio

        print(
            f"Top-{k:4d} energy ratio: "
            f"mean={np.mean(ratio):.4f}, "
            f"median={np.median(ratio):.4f}, "
            f"min={np.min(ratio):.4f}, "
            f"max={np.max(ratio):.4f}"
        )

    return H_ad, ratios


def plot_energy_curve(H):
    H_ad = np.fft.fft2(H, axes=(1, 2), norm="ortho")

    energy = np.abs(H_ad) ** 2
    flat_energy = energy.reshape(energy.shape[0], -1)

    sorted_energy = np.sort(flat_energy, axis=1)[:, ::-1]
    total_energy = np.sum(sorted_energy, axis=1, keepdims=True) + 1e-12

    cumulative_energy = np.cumsum(sorted_energy, axis=1) / total_energy

    mean_curve = np.mean(cumulative_energy, axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(mean_curve) + 1), mean_curve)
    plt.axvline(256, linestyle="--", label="256 coefficients")
    plt.axvline(512, linestyle="--", label="512 coefficients")
    plt.axvline(1024, linestyle="--", label="1024 coefficients")
    plt.title("Mean Cumulative Energy in Angular Domain")
    plt.xlabel("Number of largest coefficients kept")
    plt.ylabel("Energy ratio")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("angular_energy_concentration.png")
    plt.close()

    print("\nSaved: angular_energy_concentration.png")


def plot_sample_magnitudes(H):
    sample = H[0]
    H_ad_sample = np.fft.fft2(sample, norm="ortho")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(sample), aspect="auto")
    plt.title("Raw CSI Magnitude Sample")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(H_ad_sample), aspect="auto")
    plt.title("Angular-Domain Magnitude Sample")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("csi_raw_vs_angular_sample.png")
    plt.close()

    print("Saved: csi_raw_vs_angular_sample.png")


def coefficient_statistics(H):
    H_ad = np.fft.fft2(H, axes=(1, 2), norm="ortho")

    raw_energy = np.abs(H) ** 2
    angular_energy = np.abs(H_ad) ** 2

    raw_flat = raw_energy.reshape(raw_energy.shape[0], -1)
    angular_flat = angular_energy.reshape(angular_energy.shape[0], -1)

    raw_peak_to_mean = np.max(raw_flat, axis=1) / (np.mean(raw_flat, axis=1) + 1e-12)
    angular_peak_to_mean = np.max(angular_flat, axis=1) / (
        np.mean(angular_flat, axis=1) + 1e-12
    )

    print("\n============================================================")
    print("PEAK-TO-MEAN ENERGY RATIO")
    print("============================================================")
    print(f"Raw domain mean peak/mean     : {np.mean(raw_peak_to_mean):.4f}")
    print(f"Angular domain mean peak/mean : {np.mean(angular_peak_to_mean):.4f}")


def main():
    H = load_csi()

    basic_statistics(H)

    adjacent_correlation(H)

    angular_energy_concentration(H)

    coefficient_statistics(H)

    plot_energy_curve(H)

    plot_sample_magnitudes(H)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()