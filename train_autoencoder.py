import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


# ==================================================
# DATA LOADING
# ==================================================
def load_dataset():

    print("Loading dataset...")

    data = sio.loadmat('CSI_dataset.mat')
    H_raw = data['H_dataset']

    print("Original shape:", H_raw.shape)           # (16, 64, 10000)

    # Samples first
    H = H_raw.transpose(2, 0, 1)                    # (10000, 16, 64)

    print("Transposed shape:", H.shape)

    # Separate real and imaginary parts and stack as 2 channels
    H_real = np.real(H)                              # (10000, 16, 64)
    H_imag = np.imag(H)                              # (10000, 16, 64)

    H = np.concatenate([H_real, H_imag], axis=2)    # (10000, 16, 128)

    print("Combined shape:", H.shape)

    # Flatten to (10000, 2048)
    X = H.reshape(H.shape[0], -1)

    print("Flattened shape:", X.shape)

    # -------------------------------------------------------
    # FIX 1: Per-sample peak normalization.
    # We divide each sample by its own peak absolute value,
    # guaranteeing every sample lands in [-1, 1] — exactly
    # the range that Tanh can reconstruct at the output.
    # RMS normalization produced a range of -5.6 to +5.2
    # which Tanh cannot represent, making reconstruction
    # impossible regardless of training duration.
    # Save per-sample scale so NMSE is computed on the
    # original un-normalized values (true reconstruction error).
    # -------------------------------------------------------
    power = np.max(np.abs(X), axis=1, keepdims=True)  # (10000, 1) — per-sample peak
    X_norm = X / (power + 1e-8)                        # each sample in [-1, 1]

    print("Normalization: per-sample peak  →  range guaranteed [-1, 1]")
    print(f"  Sample peak range: {power.min():.4f} – {power.max():.4f}")
    print(f"  Normalized value range: {X_norm.min():.4f} – {X_norm.max():.4f}")

    return X_norm.astype(np.float32), power.astype(np.float32)


# ==================================================
# AUTOENCODER MODEL
# ==================================================
class Autoencoder(nn.Module):

    def __init__(self, input_dim=2048, bottleneck=64):
        """
        Args:
            input_dim  : flattened CSI dimension (16 subcarriers × 64 antennas × 2 channels = 2048)
            bottleneck : latent codeword size.
                         2048 → 64  gives CR = 32x  (matches project target)
                         2048 → 128 gives CR = 16x
        """
        super(Autoencoder, self).__init__()

        # --------------------------------------------------
        # FIX 2: Added BatchNorm after every hidden layer.
        # Without BN, deep FC networks develop internal
        # covariate shift — gradients shrink and NMSE
        # plateaus early.
        #
        # FIX 3: Switched ReLU → LeakyReLU(0.1).
        # ReLU kills neurons that receive negative input
        # (dead neuron problem), which is common with CSI
        # data that has both positive and negative values.
        #
        # FIX 4: Bottleneck has NO activation — the latent
        # representation should be unconstrained so the
        # decoder can use the full real number line.
        # --------------------------------------------------
        self.encoder = nn.Sequential(

            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),

            nn.Linear(256, bottleneck)
            # No activation — unconstrained latent space
        )

        # --------------------------------------------------
        # FIX 5: Output activation changed to Tanh.
        # The old decoder ended with no activation on the
        # last hidden layer (ReLU on second-to-last), which
        # means the final Linear output was unbounded but
        # the gradient signal was weak.
        # Tanh bounds output to [-1, 1], matching the range
        # of per-sample RMS-normalized CSI data and giving
        # the loss function a well-conditioned target space.
        # --------------------------------------------------
        self.decoder = nn.Sequential(

            nn.Linear(bottleneck, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),

            nn.Linear(1024, input_dim),
            nn.Tanh()                   # bounds output to [-1, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# ==================================================
# TRAINING
# ==================================================
def train_model():

    X, power = load_dataset()                         # X: (10000, 2048), power: (10000, 1)

    X_tensor     = torch.tensor(X)
    power_tensor = torch.tensor(power)

    full_dataset = TensorDataset(X_tensor, X_tensor, power_tensor)

    # --------------------------------------------------
    # FIX 6: Split into train / validation sets.
    # The original code trained on the full dataset with
    # no validation — the LR scheduler and early stopping
    # need a held-out set to monitor generalisation.
    # --------------------------------------------------
    n_total = len(full_dataset)
    n_val   = int(0.15 * n_total)                     # 15% validation
    n_train = n_total - n_val

    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=256, shuffle=False)

    print(f"Train samples: {n_train}  |  Val samples: {n_val}")

    model     = Autoencoder(input_dim=2048, bottleneck=64)
    criterion = nn.MSELoss()

    # --------------------------------------------------
    # FIX 7: Higher initial LR (0.001) + ReduceLROnPlateau.
    # 0.0003 is too conservative for a network starting
    # from random weights — training converges very slowly.
    # The scheduler halves LR when val loss stalls,
    # allowing fine-grained convergence later.
    # --------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # --------------------------------------------------
    # FIX 8: More epochs (200) with early stopping.
    # 100 epochs is insufficient for 32x compression —
    # the decoder needs many gradient updates to learn
    # to reconstruct from a 64-dim codeword.
    # --------------------------------------------------
    epochs        = 200
    patience      = 25           # early stopping patience
    best_val_loss = float('inf')
    patience_ctr  = 0

    print(f"\nStarting training  (input=2048, bottleneck=64, CR=32x)")
    print("=" * 60)

    train_loss_history = []
    val_loss_history   = []

    for epoch in range(epochs):

        # ── Training ──────────────────────────────────
        model.train()
        total_train_loss = 0

        for batch_x, _, _ in train_loader:

            optimizer.zero_grad()
            output = model(batch_x)
            loss   = criterion(output, batch_x)

            # ------------------------------------------
            # FIX 9: Gradient clipping prevents exploding
            # gradients in the early epochs.
            # ------------------------------------------
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # ── Validation + NMSE ─────────────────────────
        model.eval()
        total_val_loss = 0

        all_orig  = []
        all_recon = []
        all_power = []

        with torch.no_grad():
            for batch_x, _, batch_power in val_loader:
                output         = model(batch_x)
                val_loss       = criterion(output, batch_x)
                total_val_loss += val_loss.item()

                all_orig.append(batch_x.numpy())
                all_recon.append(output.numpy())
                all_power.append(batch_power.numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        # ── NMSE (de-normalized, in dB) ───────────────
        orig_np  = np.concatenate(all_orig,  axis=0)
        recon_np = np.concatenate(all_recon, axis=0)
        pow_np   = np.concatenate(all_power, axis=0)

        H_orig  = orig_np  * pow_np               # de-normalize
        H_recon = recon_np * pow_np

        error    = H_orig - H_recon
        nmse_lin = np.mean(
            np.sum(error ** 2, axis=1) / (np.sum(H_orig ** 2, axis=1) + 1e-8)
        )
        nmse_db  = 10 * np.log10(nmse_lin + 1e-12)

        scheduler.step(avg_val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:>3}/{epochs}  "
              f"Train: {avg_train_loss:.6f}  "
              f"Val: {avg_val_loss:.6f}  "
              f"NMSE: {nmse_db:+.2f} dB  "
              f"LR: {current_lr:.2e}")

        # ── Early stopping ────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_ctr  = 0
            torch.save(model.state_dict(), "autoencoder_model.pth")
            print(f"  ✓ Best model saved (val_loss={best_val_loss:.6f})")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}.")
                break

    print("\nTraining complete. Best model saved to autoencoder_model.pth")

    # ── Save per-sample power for evaluation ──────────
    np.save("csi_power.npy", power)
    print("Per-sample power saved to csi_power.npy")

    # ── Loss curves ───────────────────────────────────
    plt.figure(figsize=(9, 4))
    plt.plot(train_loss_history, label="Train Loss", linewidth=1.8)
    plt.plot(val_loss_history,   label="Val Loss",   linewidth=1.8, linestyle='--')
    plt.title("Autoencoder Training Loss  (CR = 32x)", fontsize=13)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_loss.png", dpi=150)
    plt.show()
    print("Saved: training_loss.png")


# ==================================================
# MAIN
# ==================================================
if __name__ == "__main__":
    train_model()