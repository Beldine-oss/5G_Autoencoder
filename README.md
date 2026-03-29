# AI-Enhanced Hybrid Beamforming for CSI Reduction in 5G

## Overview
This project implements an AI-based CSI compression framework using an Autoencoder to reduce feedback overhead in 5G systems, combined with hybrid beamforming for efficient MIMO transmission.

## Technologies
- Python (AI model and channel simulation)
- PyTorch (Autoencoder Training)
- NumPy (Data processing)

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the complete pipeline:
   ```bash
   python run_pipeline.py
   ```

   Or run individual steps:
   ```bash
   python generate_data.py          # Generate synthetic CSI data
   python train_autoencoder.py      # Train CSI compression autoencoder
   python hybrid_beamforming.py     # Train hybrid beamforming network
   python evaluate_performance.py   # Evaluate performance metrics
   ```

## Key Features

- **CSI Compression**: Autoencoder reduces feedback from ~16K to 64 dimensions (256x compression)
- **Hybrid Beamforming**: AI-designed beamformers using only 8 RF chains for 32 antennas
- **Spectral Efficiency**: Maintains high data rates with reduced hardware complexity
- **PyTorch Implementation**: GPU-accelerated training and inference