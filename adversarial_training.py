"""
Adversarial Training Entrypoint
================================
Wrapper script allowing `python adversarial_training.py` to be executed from the root directory.
Delegates execution to `src/risk_engine/adversarial_training.py`.
"""

import sys
import os

# Add src/risk_engine to sys.path so relative imports within risk_engine function seamlessly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'risk_engine'))

from adversarial_training import main

if __name__ == '__main__':
    main()
