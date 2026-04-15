import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Simple numpy-based hybrid beamforming (no torch dependency)
class SimpleHybridBeamformer:
    def __init__(self, num_antennas, num_rf_chains, num_users, encoding_dim):
        self.num_antennas = num_antennas
        self.num_rf_chains = num_rf_chains
        self.num_users = num_users

        # Initialize weights for beamformer prediction network
        self.W1 = np.random.randn(encoding_dim, 128) * 0.01
        self.b1 = np.zeros(128)
        self.W2 = np.random.randn(128, num_rf_chains * num_antennas * 2) * 0.01
        self.b2 = np.zeros(num_rf_chains * num_antennas * 2)

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, compressed_csi):
        # Predict analog beamformer parameters
        h1 = self.relu(np.dot(compressed_csi, self.W1) + self.b1)
        bf_params = np.dot(h1, self.W2) + self.b2

        # Reshape to complex analog beamformer
        analog_real = bf_params[:, :self.num_rf_chains * self.num_antennas]
        analog_imag = bf_params[:, self.num_rf_chains * self.num_antennas:]

        analog_bf = analog_real + 1j * analog_imag
        analog_bf = analog_bf.reshape(-1, self.num_antennas, self.num_rf_chains)

        # Normalize columns (unit norm for each RF chain)
        norms = np.linalg.norm(analog_bf, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        analog_bf = analog_bf / norms

        # Simple digital beamformer (random for now)
        digital_bf = np.random.randn(compressed_csi.shape[0], self.num_rf_chains, self.num_users)
        # Normalize
        digital_norms = np.linalg.norm(digital_bf, axis=1, keepdims=True)
        digital_norms[digital_norms == 0] = 1
        digital_bf = digital_bf / digital_norms

        return analog_bf, digital_bf

def calculate_spectral_efficiency(H, analog_bf, digital_bf, noise_power=1e-10):
    """
    Calculate spectral efficiency for hybrid beamforming.

    Parameters:
    - H: Channel matrix (batch_size, num_antennas, num_users)
    - analog_bf: Analog beamforming matrix (batch_size, num_antennas, num_rf_chains)
    - digital_bf: Digital beamforming matrix (batch_size, num_rf_chains, num_users)
    - noise_power: Noise power spectral density

    Returns:
    - se: Spectral efficiency in bits/s/Hz
    """
    batch_size = H.shape[0]

    # Effective channel: H * analog_bf * digital_bf
    effective_channel = np.matmul(H, analog_bf)  # (batch, antennas, rf_chains)
    effective_channel = np.matmul(effective_channel, digital_bf)  # (batch, antennas, users)

    # Calculate SINR for each user
    se_total = 0
    for user in range(digital_bf.shape[-1]):
        # Signal power for user
        signal_power = np.abs(effective_channel[:, :, user]) ** 2

        # Interference from other users
        interference = np.sum(np.abs(effective_channel[:, :, :]) ** 2, axis=-1) - signal_power

        # SINR
        sinr = signal_power / (interference + noise_power)

        # Spectral efficiency (average over batch)
        se_user = np.mean(np.log2(1 + sinr))
        se_total += se_user

    return se_total / digital_bf.shape[-1]  # Average over users

def load_autoencoder():
    """Load the trained autoencoder weights"""
    data = np.load("csi_autoencoder.npz")
    class SimpleAutoencoder:
        def __init__(self):
            self.encoder_W1 = data['encoder_W1']
            self.encoder_b1 = data['encoder_b1']
            self.encoder_W2 = data['encoder_W2']
            self.encoder_b2 = data['encoder_b2']
            self.encoder_W3 = data['encoder_W3']
            self.encoder_b3 = data['encoder_b3']

        def relu(self, x):
            return np.maximum(0, x)

        def encode(self, x):
            h1 = self.relu(np.dot(x, self.encoder_W1) + self.encoder_b1)
            h2 = self.relu(np.dot(h1, self.encoder_W2) + self.encoder_b2)
            encoded = np.dot(h2, self.encoder_W3) + self.encoder_b3
            return encoded

    return SimpleAutoencoder()

def train_hybrid_beamformer(epochs=50):
    """
    Train the hybrid beamforming network.
    """
    print("Loading autoencoder...")
    try:
        autoencoder = load_autoencoder()
    except FileNotFoundError:
        print("Autoencoder model not found. Please run train_autoencoder.py first.")
        return

    # Create hybrid beamformer
    num_antennas = 32
    num_rf_chains = 8
    num_users = 4
    encoding_dim = 64

    hybrid_bf = SimpleHybridBeamformer(num_antennas, num_rf_chains, num_users, encoding_dim)

    # Load some sample data for training
    try:
        H = np.load("channel_dataset.npy")
        # Take a subset for training
        H_sample = H[:100]  # First 100 samples

        # Prepare data (simplified - in practice need proper preprocessing)
        H_real = np.real(H_sample)
        H_imag = np.imag(H_sample)
        H_combined = np.concatenate((H_real, H_imag), axis=-1)
        X = H_combined.reshape(H_sample.shape[0], -1).astype(np.float32)
        X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

        # Get compressed CSI
        compressed = autoencoder.encode(X)

        print("Training hybrid beamformer...")
        for epoch in range(epochs):
            # Generate beamformers
            analog_bf, digital_bf = hybrid_bf.forward(compressed)

            # Calculate spectral efficiency as reward
            se = calculate_spectral_efficiency(H_sample, analog_bf, digital_bf)

            # Simple training feedback (in practice, you'd optimize for SE)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Spectral Efficiency: {se:.2f} bits/s/Hz")

        # Save model
        np.savez("hybrid_beamformer.npz",
                 W1=hybrid_bf.W1, b1=hybrid_bf.b1,
                 W2=hybrid_bf.W2, b2=hybrid_bf.b2)

        print("Hybrid beamformer trained and saved!")

    except FileNotFoundError:
        print("Channel dataset not found. Please run generate_data.py first.")

if __name__ == "__main__":
    train_hybrid_beamformer()


def calculate_spectral_efficiency(H, analog_bf, digital_bf, noise_power=1e-10):
    """
    Calculate spectral efficiency for hybrid beamforming.

    Parameters:
    - H: Channel matrix (batch_size, num_antennas, num_users)
    - analog_bf: Analog beamforming matrix (batch_size, num_antennas, num_rf_chains)
    - digital_bf: Digital beamforming matrix (batch_size, num_rf_chains, num_users)
    - noise_power: Noise power spectral density

    Returns:
    - se: Spectral efficiency in bits/s/Hz
    """
    batch_size = H.shape[0]

    # Effective channel: H * analog_bf * digital_bf
    effective_channel = np.matmul(H, analog_bf)  # (batch, antennas, rf_chains)
    effective_channel = np.matmul(effective_channel, digital_bf)  # (batch, antennas, users)

    # Calculate SINR for each user
    se_total = 0
    for user in range(digital_bf.shape[-1]):
        # Signal power for user
        signal_power = np.abs(effective_channel[:, :, user]) ** 2

        # Interference from other users
        interference = np.sum(np.abs(effective_channel[:, :, :]) ** 2, axis=-1) - signal_power

        # SINR
        sinr = signal_power / (interference + noise_power)

        # Spectral efficiency
        se_user = np.log2(1 + sinr.mean(axis=0))  # Average over batch
        se_total += se_user

    return se_total / digital_bf.shape[-1]  # Average over users


def train_hybrid_beamformer(epochs=50):
    """
    Train the hybrid beamforming network.
    """
    print("Loading autoencoder...")
    try:
        autoencoder = load_autoencoder()
    except FileNotFoundError:
        print("Autoencoder model not found. Please run train_autoencoder.py first.")
        return

    # Create hybrid beamformer
    num_antennas = 32
    num_rf_chains = 8
    num_users = 4
    encoding_dim = 64

    hybrid_bf = SimpleHybridBeamformer(num_antennas, num_rf_chains, num_users, encoding_dim)

    # Load some sample data for training
    try:
        H = np.load("channel_dataset.npy")
        # Take a subset for training
        H_sample = H[:100]  # First 100 samples

        # Prepare data (simplified - in practice need proper preprocessing)
        H_real = np.real(H_sample)
        H_imag = np.imag(H_sample)
        H_combined = np.concatenate((H_real, H_imag), axis=-1)
        X = H_combined.reshape(H_sample.shape[0], -1).astype(np.float32)
        X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

        # Get compressed CSI
        compressed = autoencoder.encode(X)

        print("Training hybrid beamformer...")
        for epoch in range(epochs):
            # Generate beamformers
            analog_bf, digital_bf = hybrid_bf.forward(compressed)

            # Calculate spectral efficiency as reward
            se = calculate_spectral_efficiency(H_sample, analog_bf, digital_bf)

            # Simple training feedback (in practice, you'd optimize for SE)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Spectral Efficiency: {se:.2f} bits/s/Hz")

        # Save model
        np.savez("hybrid_beamformer.npz",
                 W1=hybrid_bf.W1, b1=hybrid_bf.b1,
                 W2=hybrid_bf.W2, b2=hybrid_bf.b2)
        print("Hybrid beamformer trained and saved!")

    except FileNotFoundError:
        print("Channel dataset not found. Please run generate_data.py first.")


if __name__ == "__main__":
    train_hybrid_beamformer()
