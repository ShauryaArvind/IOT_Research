"""
ONNX Model Export & Verification Pipeline
=========================================
Part of Phase 11: Physical Edge Hardware Deployment & Benchmarking.

Exports the trained PyTorch models of one run (baseline classifier, robust
classifier, denoising autoencoder) to ONNX, and writes scaler_params.json
from the run's saved preprocessor. Nothing is refitted here, so the
edge runtime standardises exactly as training did.

    python src/risk_engine/export_onnx.py --config legacy_balanced --seed 42 --robust-defense pgd_at
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

import argparse
import json
import numpy as np
import torch
import onnx
import onnxruntime as ort


# ---------------------------------------------------------------------------
# OUTPUT PATHS (read by edge_inference_engine.py)
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
BASELINE_ONNX = os.path.join(MODELS_DIR, 'network_risk_classifier.onnx')
ADV_ONNX = os.path.join(MODELS_DIR, 'network_risk_classifier_adv.onnx')
DAE_ONNX = os.path.join(MODELS_DIR, 'denoising_autoencoder.onnx')
SCALER_JSON = os.path.join(MODELS_DIR, 'scaler_params.json')


def pprint(*args, **kwargs):
    """Print wrapper forcing immediate stdout flush for real-time logging."""
    print(*args, **kwargs)
    sys.stdout.flush()


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
    return onnx_size_kb(onnx_path)


def onnx_size_kb(onnx_path):
    """Size of the ONNX graph plus its external weight file (<name>.onnx.data), if any.
    The legacy footprint numbers counted only the graph file and left out the weights."""
    total = os.path.getsize(onnx_path)
    if os.path.exists(onnx_path + '.data'):
        total += os.path.getsize(onnx_path + '.data')
    return total / 1024.0


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


def _export(name, model, sample_input, onnx_path, output_name):
    size_kb = export_model_to_onnx(model, sample_input, onnx_path, 'flow_features', output_name)
    diff = verify_onnx_equivalence(model, onnx_path, sample_input, 'flow_features')
    status = 'PASSED' if diff < 1e-4 else 'CHECK'
    pprint(f"  [+] {name} -> {onnx_path} ({size_kb:.1f} KB, incl. external .data if any)")
    pprint(f"      max |PyTorch - ONNX| on 256 test rows: {diff:.2e} ({status})")


def export_all_models(argv=None):
    from denoising_autoencoder import load_dae
    from training import DEFAULT_CONFIG, load_run, run_dir, splits_for_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=DEFAULT_CONFIG)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--robust-defense', default='pgd_at', choices=['pgd_at', 'trades', 'fgsm_at'])
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)

    pprint("=" * 80)
    pprint("PHASE 11: ONNX MODEL EXPORT & NUMERICAL EQUIVALENCE VERIFICATION")
    pprint("=" * 80)

    base, meta, prep = load_run(run_dir(a.config, 'none', a.seed))
    with open(SCALER_JSON, 'w') as f:
        json.dump(prep.to_json_dict(), f, indent=2)
    pprint(f"  [+] Preprocessor ({meta['scaler_path']}) written to {SCALER_JSON}")

    splits = splits_for_config(a.config, a.seed, skip_resampling=True, verbose=False)
    sample_input = torch.FloatTensor(splits.X_test[:256])

    pprint("\n[1/3] Baseline classifier")
    _export('baseline', base, sample_input, BASELINE_ONNX, 'class_logits')

    pprint(f"\n[2/3] Robust classifier ({a.robust_defense})")
    rd = run_dir(a.config, a.robust_defense, a.seed)
    if (rd / 'classifier.pth').exists():
        _export(a.robust_defense, load_run(rd)[0], sample_input, ADV_ONNX, 'class_logits')
    else:
        pprint(f"  [-] no run at {rd}; skipped")

    pprint("\n[3/3] Denoising autoencoder")
    try:
        dae, _ = load_dae(a.config, a.seed)
        _export('dae', dae, sample_input, DAE_ONNX, 'reconstructed_features')
    except FileNotFoundError:
        pprint("  [-] no DAE run; skipped")


if __name__ == '__main__':
    export_all_models()
