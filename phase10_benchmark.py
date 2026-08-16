"""
Phase 10 Benchmark Entrypoint
=============================
Wrapper script allowing `python phase10_benchmark.py` to be executed from the root directory.
Delegates execution to `src/risk_engine/evaluate_phase10.py`.
"""

import sys
import os

# Add src/risk_engine to sys.path so relative imports within risk_engine function seamlessly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'risk_engine'))

from evaluate_phase10 import evaluate_phase10_suite

if __name__ == '__main__':
    evaluate_phase10_suite()
