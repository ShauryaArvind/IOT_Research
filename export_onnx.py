"""
ONNX Export Entrypoint
======================
Wrapper script allowing `python export_onnx.py` to be executed from the root directory.
Delegates execution to `src/risk_engine/export_onnx.py`.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'risk_engine'))

from export_onnx import export_all_models

if __name__ == '__main__':
    export_all_models()
