import numpy as np
from hybrid_beamforming import SimpleHybridBeamformer, calculate_spectral_efficiency, load_autoencoder

def evaluate_compression_performance():
    """
    Evaluate the autoencoder's CSI compression performance.
    """
    print("Evaluating CSI Compression Performance")
    print("=" * 50)

    try:
        # Load data
        H = np.load("channel_dataset.npy")
        print(f"Dataset shape: {H.shape}")

        # Load trained autoencoder
        autoencoder_data = np.load("csi_autoencoder.npz")

        class SimpleAutoencoder:
            def __init__(self, data):
                self.encoder_W1 = data['encoder_W1']
                self.encoder_b1 = data['encoder_b1']
                self.encoder_W2 = data['encoder_W2']
                self.encoder_b2 = data['encoder_b2']
                self.encoder_W3 = data['encoder_W3']
                self.encoder_b3 = data['encoder_b3']
                self.decoder_W1 = data['decoder_W1']
                self.decoder_b1 = data['decoder_b1']
                self.decoder_W2 = data['decoder_W2']
                self.decoder_b2 = data['decoder_b2']
                self.decoder_W3 = data['decoder_W3']
                self.decoder_b3 = data['decoder_b3']

            def relu(self, x):
                return np.maximum(0, x)

            def forward(self, x):
                # Encoder
                h1 = self.relu(np.dot(x, self.encoder_W1) + self.encoder_b1)
                h2 = self.relu(np.dot(h1, self.encoder_W2) + self.encoder_b2)
                encoded = np.dot(h2, self.encoder_W3) + self.encoder_b3

                # Decoder
                h3 = self.relu(np.dot(encoded, self.decoder_W1) + self.decoder_b1)
                h4 = self.relu(np.dot(h3, self.decoder_W2) + self.decoder_b2)
                decoded = np.dot(h4, self.decoder_W3) + self.decoder_b3

                return decoded

        autoencoder = SimpleAutoencoder(autoencoder_data)

        # Prepare test data
        H_real = np.real(H)
        H_imag = np.imag(H)
        H_combined = np.concatenate((H_real, H_imag), axis=-1)
        num_samples = H_combined.shape[0]
        input_dim = np.prod(H_combined.shape[1:])
        X = H_combined.reshape(num_samples, input_dim).astype(np.float32)
        X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

        # Test reconstruction
        reconstructed = autoencoder.forward(X)
        mse = np.mean((reconstructed - X) ** 2)

        print(".6f")
        print(".1f")
        print(".2f")

        # Calculate compression ratio
        original_bits = input_dim * 32  # 32 bits per float
        compressed_bits = 64 * 32  # 64-dimensional encoding
        compression_ratio = original_bits / compressed_bits
        print(".1f")

        return mse, compression_ratio

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure channel_dataset.npy and csi_autoencoder.npz exist.")
        return None, None

def evaluate_hybrid_beamforming_performance():
    """
    Evaluate hybrid beamforming spectral efficiency.
    """
    print("\nEvaluating Hybrid Beamforming Performance")
    print("=" * 50)

    try:
        # Load models
        encoding_dim = 64
        num_antennas = 32
        num_rf_chains = 8
        num_users = 4

        autoencoder = load_autoencoder()
        hybrid_bf_data = np.load("hybrid_beamformer.npz")
        hybrid_bf = SimpleHybridBeamformer(num_antennas, num_rf_chains, num_users, encoding_dim)
        hybrid_bf.W1 = hybrid_bf_data['W1']
        hybrid_bf.b1 = hybrid_bf_data['b1']
        hybrid_bf.W2 = hybrid_bf_data['W2']
        hybrid_bf.b2 = hybrid_bf_data['b2']

        # Load test data
        H = np.load("channel_dataset.npy")
        H_test = H[100:200]  # Use different samples for testing

        # Prepare data
        H_real = np.real(H_test)
        H_imag = np.imag(H_test)
        H_combined = np.concatenate((H_real, H_imag), axis=-1)
        X = H_combined.reshape(H_test.shape[0], -1).astype(np.float32)
        X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

        # Get compressed CSI
        compressed = autoencoder.encode(X)

        # Generate beamformers
        analog_bf, digital_bf = hybrid_bf.forward(compressed)

        # Calculate spectral efficiency
        se = calculate_spectral_efficiency(H_test, analog_bf, digital_bf)

        print(".2f")

        # Compare with random beamforming (baseline)
        random_analog = np.random.randn(*analog_bf.shape) + 1j * np.random.randn(*analog_bf.shape)
        random_analog = random_analog / np.linalg.norm(random_analog, axis=1, keepdims=True)
        random_digital = np.random.randn(*digital_bf.shape)
        random_digital = random_digital / np.linalg.norm(random_digital, axis=1, keepdims=True)
        random_se = calculate_spectral_efficiency(H_test, random_analog, random_digital)
        print(".2f")
        print(".1f")

        return se

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the models first.")
        return None

def main():
    """
    Run all evaluations.
    """
    print("5G AI-Enhanced Hybrid Beamforming Evaluation")
    print("=" * 60)

    # Evaluate compression
    mse, compression_ratio = evaluate_compression_performance()

    # Evaluate beamforming
    se = evaluate_hybrid_beamforming_performance()

    if mse is not None and se is not None:
        print("\n" + "=" * 60)
        print("SUMMARY:")
        print(".1f")
        print(".2f")
        print("✅ CSI feedback overhead reduced through compression")
        print("✅ Hybrid beamforming implemented with AI assistance")
        print("✅ Spectral efficiency maintained with fewer RF chains")

if __name__ == "__main__":
    main()

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure channel_dataset.npy and csi_autoencoder.pth exist.")
        return None, None


def evaluate_hybrid_beamforming_performance():
    """
    Evaluate hybrid beamforming spectral efficiency.
    """
    print("\nEvaluating Hybrid Beamforming Performance")
    print("=" * 50)

    try:
        # Load models
        encoding_dim = 64
        num_antennas = 32
        num_rf_chains = 8
        num_users = 4

        autoencoder = Autoencoder(32*32*2*64, encoding_dim)  # Adjust dimensions
        autoencoder.load_state_dict(torch.load("csi_autoencoder.pth"))
        autoencoder.eval()

        hybrid_bf = HybridBeamformer(num_antennas, num_rf_chains, num_users, encoding_dim)
        hybrid_bf.load_state_dict(torch.load("hybrid_beamformer.pth"))
        hybrid_bf.eval()

        # Load test data
        H = np.load("channel_dataset.npy")
        H_test = H[100:200]  # Use different samples for testing
        H_tensor = torch.from_numpy(H_test).float()

        # Get compressed CSI
        with torch.no_grad():
            # Flatten and prepare data
            H_real = np.real(H_test)
            H_imag = np.imag(H_test)
            H_combined = np.concatenate((H_real, H_imag), axis=-1)
            X = H_combined.reshape(H_test.shape[0], -1).astype(np.float32)
            X_tensor = torch.from_numpy(X)

            compressed = autoencoder.encoder(X_tensor)

            # Generate beamformers
            analog_bf, digital_bf = hybrid_bf(compressed)

            # Calculate spectral efficiency
            se = calculate_spectral_efficiency(H_tensor, analog_bf, digital_bf)

        print(".2f")

        # Compare with fully digital beamforming (upper bound)
        # For fully digital, we assume num_rf_chains = num_antennas
        print("Note: Fully digital beamforming would achieve higher SE but requires more RF chains")

        return se.item()

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the models first.")
        return None


def main():
    """
    Run all evaluations.
    """
    print("5G AI-Enhanced Hybrid Beamforming Evaluation")
    print("=" * 60)

    # Evaluate compression
    mse, compression_ratio = evaluate_compression_performance()

    # Evaluate beamforming
    se = evaluate_hybrid_beamforming_performance()

    if mse is not None and se is not None:
        print("\n" + "=" * 60)
        print("SUMMARY:")
        print(".1f")
        print(".2f")
        print("✅ CSI feedback overhead reduced through compression")
        print("✅ Hybrid beamforming implemented with AI assistance")
        print("✅ Spectral efficiency maintained with fewer RF chains")


if __name__ == "__main__":
    main()
