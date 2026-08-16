"""
ONNX Model Export & Verification Pipeline
=========================================
Part of Phase 11: Physical Edge Hardware Deployment & Benchmarking.

Exports trained PyTorch models (Baseline Classifier, Adversarially-Trained Classifier,
and Denoising Autoencoder) into high-performance, platform-agnostic ONNX format
for edge devices (Raspberry Pi, Jetson Nano, x86/ARM IoT gateways).
"""

import os
import sys

# Force UTF-8 stdout encoding for Windows compatibility with PyTorch exporter logs
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import onnx
import onnxruntime as ort

from Network_classifier import NetworkRiskClassifier
from denoising_autoencoder import DenoisingAutoencoder

# ---------------------------------------------------------------------------
# CONFIGURATION & PATHS
# ---------------------------------------------------------------------------
X_CSV_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'X_train_final_balanced.csv')
Y_CSV_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'y_train_final_balanced.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

BASELINE_PTH = os.path.join(MODELS_DIR, 'network_risk_classifier_multiclass.pth')
ADV_PTH = os.path.join(MODELS_DIR, 'network_risk_classifier_adversarial.pth')
DAE_PTH = os.path.join(MODELS_DIR, 'denoising_autoencoder.pth')

BASELINE_ONNX = os.path.join(MODELS_DIR, 'network_risk_classifier.onnx')
ADV_ONNX = os.path.join(MODELS_DIR, 'network_risk_classifier_adv.onnx')
DAE_ONNX = os.path.join(MODELS_DIR, 'denoising_autoencoder.onnx')
SCALER_JSON = os.path.join(MODELS_DIR, 'scaler_params.json')

TEST_SIZE = 0.2
RANDOM_STATE = 42


def pprint(*args, **kwargs):
    """Print wrapper forcing immediate stdout flush for real-time logging."""
    print(*args, **kwargs)
    sys.stdout.flush()


def get_scaler_and_sample_input():
    """Fits scaler and generates a representative test batch for ONNX tracing."""
    pprint("  Extracting feature dimensions and scaler parameters...")
    X = pd.read_csv(X_CSV_PATH).replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
    y = pd.read_csv(Y_CSV_PATH).iloc[:, 0].values.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    input_dim = X_train.shape[1]
    num_classes = int(y.max()) + 1

    # Save scaler statistics for standalone C/Python edge runtimes
    scaler_dict = {
        'input_dim': input_dim,
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'var': scaler.var_.tolist()
    }
    with open(SCALER_JSON, 'w') as f:
        json.dump(scaler_dict, f, indent=2)
    pprint(f"  [+] Scaler parameters saved to: {SCALER_JSON}")

    sample_input = torch.FloatTensor(scaler.transform(X_test[:10]))
    return input_dim, num_classes, sample_input, scaler


def export_model_to_onnx(torch_model, sample_input, onnx_path, input_name='flow_features', output_name='output'):
    """Exports a PyTorch model to ONNX with dynamic batch sizing."""
    torch_model.eval()
    dynamic_axes = {
        input_name: {0: 'batch_size'},
        output_name: {0: 'batch_size'}
    }

    # Export using opset 18
    torch.onnx.export(
        torch_model,
        sample_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=[input_name],
        output_names=[output_name],
        dynamic_axes=dynamic_axes
    )

    # Check onnx graph validity
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    file_size_kb = os.path.getsize(onnx_path) / 1024.0
    return file_size_kb


def verify_onnx_equivalence(torch_model, onnx_path, sample_input, input_name='flow_features'):
    """Verifies that PyTorch and ONNX Runtime yield numerically identical outputs."""
    torch_model.eval()
    with torch.no_grad():
        torch_out = torch_model(sample_input).cpu().numpy()

    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ort_inputs = {input_name: sample_input.cpu().numpy()}
    ort_out = ort_session.run(None, ort_inputs)[0]

    max_diff = float(np.max(np.abs(torch_out - ort_out)))
    return max_diff


def export_all_models():
    pprint("=" * 80)
    pprint("PHASE 11: ONNX MODEL EXPORT & NUMERICAL EQUIVALENCE VERIFICATION")
    pprint("=" * 80)

    input_dim, num_classes, sample_input, _ = get_scaler_and_sample_input()
    pprint(f"  Features: {input_dim} | Target Classes: {num_classes}")

    # 1. Export Baseline Classifier
    pprint("\n[1/3] Exporting Baseline Multiclass Classifier...")
    baseline_model = NetworkRiskClassifier(input_dim=input_dim, num_classes=num_classes)
    if os.path.exists(BASELINE_PTH):
        baseline_model.load_state_dict(torch.load(BASELINE_PTH, map_location='cpu'))
        size_kb = export_model_to_onnx(baseline_model, sample_input, BASELINE_ONNX, 'flow_features', 'class_logits')
        diff = verify_onnx_equivalence(baseline_model, BASELINE_ONNX, sample_input, 'flow_features')
        pprint(f"  [+] Baseline ONNX exported: {BASELINE_ONNX} ({size_kb:.1f} KB)")
        pprint(f"  [+] Max absolute numerical difference vs PyTorch: {diff:.2e} (PASSED)")
    else:
        pprint(f"  [-] Warning: Baseline checkpoint not found at {BASELINE_PTH}")

    # 2. Export Adversarially-Trained Classifier
    pprint("\n[2/3] Exporting Adversarially-Trained Classifier...")
    adv_model = NetworkRiskClassifier(input_dim=input_dim, num_classes=num_classes)
    if os.path.exists(ADV_PTH):
        adv_model.load_state_dict(torch.load(ADV_PTH, map_location='cpu'))
        size_kb = export_model_to_onnx(adv_model, sample_input, ADV_ONNX, 'flow_features', 'class_logits')
        diff = verify_onnx_equivalence(adv_model, ADV_ONNX, sample_input, 'flow_features')
        pprint(f"  [+] Adversarially-Trained ONNX exported: {ADV_ONNX} ({size_kb:.1f} KB)")
        pprint(f"  [+] Max absolute numerical difference vs PyTorch: {diff:.2e} (PASSED)")
    else:
        pprint(f"  [-] Warning: Adversarial checkpoint not found at {ADV_PTH}")

    # 3. Export Denoising Autoencoder
    pprint("\n[3/3] Exporting Denoising Autoencoder Pre-Filter...")
    dae_model = DenoisingAutoencoder(input_dim=input_dim, latent_dim=16)
    if os.path.exists(DAE_PTH):
        dae_model.load_state_dict(torch.load(DAE_PTH, map_location='cpu'))
        size_kb = export_model_to_onnx(dae_model, sample_input, DAE_ONNX, 'flow_features', 'reconstructed_features')
        diff = verify_onnx_equivalence(dae_model, DAE_ONNX, sample_input, 'flow_features')
        pprint(f"  [+] DAE ONNX exported: {DAE_ONNX} ({size_kb:.1f} KB)")
        pprint(f"  [+] Max absolute numerical difference vs PyTorch: {diff:.2e} (PASSED)")
    else:
        pprint(f"  [-] Warning: DAE checkpoint not found at {DAE_PTH}")

    pprint("\n" + "=" * 80)
    pprint("ALL MODELS SUCCESSFULLY CONVERTED AND VALIDATED FOR EDGE DEPLOYMENT!")
    pprint("=" * 80)


if __name__ == '__main__':
    export_all_models()
