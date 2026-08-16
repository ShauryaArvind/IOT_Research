"""
Denoising Autoencoder Entrypoint
================================
Wrapper script allowing `python denoising_autoencoder.py` to be executed from the root directory.
Delegates execution to `src/risk_engine/denoising_autoencoder.py`.
"""

import sys
import os

# Add src/risk_engine to sys.path so relative imports within risk_engine function seamlessly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'risk_engine'))

from denoising_autoencoder import train_denoising_autoencoder

if __name__ == '__main__':
    train_denoising_autoencoder()
