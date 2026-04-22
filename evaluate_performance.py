import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io as sio

from train_autoencoder import Autoencoder


# ==================================================
# DATA LOADING
# Must match train_autoencoder.py exactly.
# ==================================================
def load_dataset():
    """
    Returns
    -------
    X_norm : np.ndarray  (10000, 2048)  — per-sample RMS normalized
    H_orig : np.ndarray  (10000, 2048)  — raw (de-normalized) for NMSE
    power  : np.ndarray  (10000, 1)     — per-sample RMS scale factor
    """
    print("Loading dataset...")

    data  = sio.loadmat('CSI_dataset.mat')
    H_raw = data['H_dataset']

    print("Original shape:", H_raw.shape)           # (16, 64, 10000)

    H = H_raw.transpose(2, 0, 1)                    # (10000, 16, 64)

    print("Transposed shape:", H.shape)

    H_real = np.real(H)
    H_imag = np.imag(H)

    H = np.concatenate([H_real, H_imag], axis=2)    # (10000, 16, 128)

    print("Combined shape:", H.shape)

    X = H.reshape(H.shape[0], -1)                   # (10000, 2048)

    print("Flattened shape:", X.shape)

    # --------------------------------------------------
    # FIX 1 (eval): Per-sample RMS normalization.
    # The original eval script used z-score normalization
    # (global mean/std) while the training script used
    # global max — they were inconsistent with each other.
    # Both scripts now use the same per-sample RMS method.
    #
    # We also load the saved power array from training so
    # that NMSE is computed on the original scale, not the
    # normalized scale (which would give an artificially
    # lower number).
    # --------------------------------------------------
    try:
        power = np.load("csi_power.npy")             # saved during training
        print("Loaded per-sample power from csi_power.npy")
    except FileNotFoundError:
        print("csi_power.npy not found — recomputing from data.")
        power = np.sqrt(np.mean(X ** 2, axis=1, keepdims=True))

    X_norm = X / (power + 1e-8)
    H_orig = X                                       # raw, un-normalized

    print(f"  Normalized range: {X_norm.min():.4f} – {X_norm.max():.4f}")

    return X_norm.astype(np.float32), H_orig.astype(np.float32), power.astype(np.float32)


# ==================================================
# NMSE METRIC  (always on de-normalized values)
# ==================================================
def compute_nmse_db(H_orig, H_recon):
    """
    Parameters
    ----------
    H_orig  : (N, D) — original channel vectors, original scale
    H_recon : (N, D) — reconstructed channel vectors, original scale

    Returns
    -------
    nmse_db : scalar NMSE in dB
    """
    # --------------------------------------------------
    # FIX 2 (eval): The original function computed NMSE
    # over 4D arrays (N, C, H, W) suitable for CNN output,
    # but the model outputs flat (N, 2048) vectors.
    # This version works correctly for flat tensors.
    # --------------------------------------------------
    error    = H_orig - H_recon                           # (N, D)
    num      = np.sum(error  ** 2, axis=1)                # (N,)
    den      = np.sum(H_orig ** 2, axis=1) + 1e-8         # (N,)
    nmse_lin = np.mean(num / den)
    nmse_db  = 10 * np.log10(nmse_lin + 1e-12)
    return nmse_db


# ==================================================
# LOAD TRAINED MODEL
# ==================================================
def load_model():
    model = Autoencoder(input_dim=2048, bottleneck=64)
    model.load_state_dict(
        torch.load("autoencoder_model.pth", map_location="cpu")
    )
    model.eval()
    return model


# ==================================================
# PERFORMANCE EVALUATION
# ==================================================
def evaluate_performance():

    print("\n5G AI-Enhanced CSI Compression — Evaluation")
    print("=" * 60)

    X_norm, H_orig, power = load_dataset()

    model = load_model()

    X_tensor = torch.tensor(X_norm)

    print("Input tensor shape:", X_tensor.shape)       # (10000, 2048)

    with torch.no_grad():
        recon_norm = model(X_tensor).numpy()            # (10000, 2048) — normalized scale

    # --------------------------------------------------
    # FIX 3 (eval): De-normalize reconstructed output
    # before computing NMSE.
    # The model outputs values in [-1, 1] (Tanh).
    # Multiplying by power restores the original channel
    # scale so the NMSE reflects real reconstruction error.
    # --------------------------------------------------
    H_recon = recon_norm * power                        # de-normalize

    # ── Metrics ───────────────────────────────────────
    mse     = np.mean((H_orig - H_recon) ** 2)
    nmse_db = compute_nmse_db(H_orig, H_recon)

    original_dim   = 16 * 64 * 2                        # 2048 (real + imag)
    compressed_dim = 64                                  # bottleneck size
    cr             = original_dim / compressed_dim       # 32x

    # Cosine similarity (average over samples) — bonus metric
    dot   = np.sum(H_orig * H_recon, axis=1)
    norm1 = np.linalg.norm(H_orig,  axis=1) + 1e-8
    norm2 = np.linalg.norm(H_recon, axis=1) + 1e-8
    cos_sim = np.mean(dot / (norm1 * norm2))

    print("\nEvaluation Results")
    print("=" * 40)
    print(f"MSE               : {mse:.6f}")
    print(f"NMSE              : {nmse_db:+.2f} dB")
    print(f"Cosine Similarity : {cos_sim:.4f}  (1.0 = perfect)")
    print(f"Compression Ratio : {cr:.0f}x  "
          f"({original_dim} → {compressed_dim} dims)")
    print(f"\nTarget range for CR=32x: –8 to –12 dB (deep FC)")

    return X_norm, H_orig, H_recon, power


# ==================================================
# VISUALIZATION
# ==================================================
def visualize_reconstruction(X_norm, H_orig, H_recon, power):

    print("\nVisualizing CSI reconstruction...")

    # Reshape to (N, 16, 128) for heatmap display
    # 128 = 64 antennas × 2 (real stacked alongside imag)
    H_orig_2d  = H_orig.reshape(-1, 16, 128)
    H_recon_2d = H_recon.reshape(-1, 16, 128)

    # ── Panel 1: Heatmap comparison for 1 sample ──────
    sample = 0

    orig_real  = H_orig_2d[sample, :, :64]     # (16, 64) real part
    recon_real = H_recon_2d[sample, :, :64]    # (16, 64) real part
    error_map  = np.abs(orig_real - recon_real)

    # --------------------------------------------------
    # FIX 4 (eval): Added error heatmap as a third panel.
    # Comparing only original vs reconstructed visually
    # is hard; the error map immediately shows which
    # subcarrier/antenna indices are poorly reconstructed.
    # --------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    vmin = min(orig_real.min(), recon_real.min())
    vmax = max(orig_real.max(), recon_real.max())

    im0 = axes[0].imshow(orig_real,  aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis')
    axes[0].set_title(f"Original CSI — Real Part\n(sample {sample})", fontsize=11)
    axes[0].set_xlabel("Antenna index")
    axes[0].set_ylabel("Subcarrier index")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(recon_real, aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis')
    axes[1].set_title(f"Reconstructed CSI — Real Part\n(CR = 32x)", fontsize=11)
    axes[1].set_xlabel("Antenna index")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(error_map, aspect='auto', cmap='hot')
    axes[2].set_title("Absolute Error\n(lower = better)", fontsize=11)
    axes[2].set_xlabel("Antenna index")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig("csi_reconstruction.png", dpi=150)
    plt.show()
    print("Saved: csi_reconstruction.png")

    # ── Panel 2: NMSE distribution over all samples ───
    error_all   = H_orig - H_recon
    num_all     = np.sum(error_all ** 2, axis=1)
    den_all     = np.sum(H_orig    ** 2, axis=1) + 1e-8
    nmse_per_sample_db = 10 * np.log10(num_all / den_all + 1e-12)

    plt.figure(figsize=(7, 4))
    plt.hist(nmse_per_sample_db, bins=60, color='steelblue', edgecolor='white', linewidth=0.5)
    plt.axvline(np.mean(nmse_per_sample_db), color='red', linestyle='--',
                label=f"Mean = {np.mean(nmse_per_sample_db):.2f} dB")
    plt.title("Per-Sample NMSE Distribution  (CR = 32x)", fontsize=12)
    plt.xlabel("NMSE (dB)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("nmse_distribution.png", dpi=150)
    plt.show()
    print("Saved: nmse_distribution.png")

    # ── Panel 3: Real vs reconstructed amplitude profile ─
    antenna_idx = 0
    plt.figure(figsize=(9, 4))
    plt.plot(orig_real[:, antenna_idx],  label="Original",      linewidth=1.8)
    plt.plot(recon_real[:, antenna_idx], label="Reconstructed", linewidth=1.8, linestyle='--')
    plt.title(f"CSI Amplitude — Antenna {antenna_idx}  (all subcarriers, sample {sample})", fontsize=11)
    plt.xlabel("Subcarrier index")
    plt.ylabel("Channel amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("csi_profile.png", dpi=150)
    plt.show()
    print("Saved: csi_profile.png")


# ==================================================
# MAIN
# ==================================================
def main():

    X_norm, H_orig, H_recon, power = evaluate_performance()

    visualize_reconstruction(X_norm, H_orig, H_recon, power)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()