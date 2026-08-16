"""
Edge Inference Engine Entrypoint
================================
Wrapper script allowing `python edge_inference_engine.py` to be executed from the root directory.
Delegates execution to `src/risk_engine/edge_inference_engine.py`.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'risk_engine'))

from edge_inference_engine import run_edge_smoke_test

if __name__ == '__main__':
    run_edge_smoke_test()
