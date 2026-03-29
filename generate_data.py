import numpy as np


def generate_csi_dataset(num_samples=1000, num_antennas=64, num_subcarriers=100):
    """
    Generate synthetic CSI dataset for 5G MIMO systems.

    Parameters:
    - num_samples: Number of channel realizations
    - num_antennas: Number of antennas (e.g., 64 for massive MIMO)
    - num_subcarriers: Number of subcarriers (frequency domain)

    Returns:
    - H: Complex channel matrix of shape (num_samples, num_subcarriers, num_antennas, num_antennas)
    """

    # For simplicity, generate Rayleigh fading channels
    # In practice, this would be more sophisticated with path loss, shadowing, etc.

    H = np.zeros((num_samples, num_subcarriers, num_antennas, num_antennas), dtype=complex)

    for sample in range(num_samples):
        for subcarrier in range(num_subcarriers):
            # Generate complex Gaussian channel matrix
            real_part = np.random.normal(0, 1, (num_antennas, num_antennas))
            imag_part = np.random.normal(0, 1, (num_antennas, num_antennas))
            H[sample, subcarrier] = (real_part + 1j * imag_part) / np.sqrt(2)

    return H


if __name__ == "__main__":
    # Generate dataset
    print("Generating CSI dataset...")
    H = generate_csi_dataset(num_samples=1000, num_antennas=16, num_subcarriers=32)

    # Save as numpy file (instead of .mat)
    np.save("channel_dataset.npy", H)
    print(f"Dataset saved with shape: {H.shape}")
    print("CSI dataset generation complete!")
    print("CSI dataset generation complete!")
