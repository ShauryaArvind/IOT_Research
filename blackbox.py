"""
Black-Box Adversarial Attack Entrypoint
=======================================
Wrapper script allowing `python blackbox.py` to be executed from the root directory.
Delegates execution to `src/risk_engine/blackbox.py`.
"""

import sys
import os

# Add src/risk_engine to sys.path so relative imports within risk_engine function seamlessly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'risk_engine'))

from blackbox import main

if __name__ == '__main__':
    main()
