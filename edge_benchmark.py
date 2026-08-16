"""
Edge Benchmark Entrypoint
=========================
Wrapper script allowing `python edge_benchmark.py` to be executed from the root directory.
Delegates execution to `src/risk_engine/edge_benchmark.py`.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'risk_engine'))

from edge_benchmark import execute_edge_benchmark

if __name__ == '__main__':
    execute_edge_benchmark()
