#!/usr/bin/env python3
"""
Main script to run the complete 5G AI-Enhanced Hybrid Beamforming pipeline.
"""

import os
import sys


def main():
    print("5G AI-Enhanced Hybrid Beamforming Pipeline")
    print("=" * 50)

    steps = [
        ("Generate CSI Dataset", "python generate_data.py"),
        ("Train Autoencoder", "python train_autoencoder.py"),
        ("Train Hybrid Beamformer", "python hybrid_beamforming.py"),
        ("Evaluate Performance", "python evaluate_performance.py")
    ]

    for step_name, command in steps:
        print(f"\n🔄 {step_name}...")
        try:
            exit_code = os.system(command)
            if exit_code != 0:
                print(f"❌ {step_name} failed with exit code {exit_code}")
                sys.exit(1)
            print(f"✅ {step_name} completed successfully")
        except Exception as e:
            print(f"❌ {step_name} failed: {e}")
            sys.exit(1)

    print("\n🎉 All objectives implemented successfully!")
    print("📊 Check evaluate_performance.py output for results")


if __name__ == "__main__":
    main()
