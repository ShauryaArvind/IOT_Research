"""
Edge Inference Engine for IoT Intrusion Detection & Zero-Trust Verification
===========================================================================
Part of Phase 11: Physical Edge Hardware Deployment & Benchmarking.

Provides a production-grade, low-latency, modular runtime engine designed
for deployment on IoT edge gateways (Raspberry Pi, Jetson Nano, ARM/x86 gateways).

Pipeline:
  1. Lightweight Preprocessing: In-memory feature standardization via `scaler_params.json`.
  2. DAE Feature Sanitization: Cleans incoming adversarial flow vectors using ONNX runtime.
  3. Multiclass Neural Classification: Computes logits and class probabilities via ONNX.
  4. Context-Aware Zero-Trust Policy: Evaluates prioritized rules for instant ALLOW / DENY.
"""

import os
import sys

# Force UTF-8 stdout encoding for Windows compatibility
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import onnxruntime as ort

from zero_trust_engine import ZeroTrustEngine, AccessDecision

# ---------------------------------------------------------------------------
# PATHS & ATTACK TAXONOMY
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
ADV_ONNX_PATH = os.path.join(MODELS_DIR, 'network_risk_classifier_adv.onnx')
BASELINE_ONNX_PATH = os.path.join(MODELS_DIR, 'network_risk_classifier.onnx')
DAE_ONNX_PATH = os.path.join(MODELS_DIR, 'denoising_autoencoder.onnx')
SCALER_JSON_PATH = os.path.join(MODELS_DIR, 'scaler_params.json')

ATTACK_TAXONOMY = {
    0: "Benign Traffic",
    1: "DoS / DDoS Flood",
    2: "PortScan Reconnaissance",
    3: "Botnet Communication (C2)",
    4: "Brute Force Credentials",
    5: "Web Attack / Injection",
    6: "Infiltration / Lateral Movement",
    7: "Heartbleed / Exploit",
    8: "Volumetric Probe",
    9: "Advanced Persistent Threat (APT)"
}


@dataclass
class EdgeInferenceResult:
    """Encapsulates the full edge detection and defense decision for a network flow."""
    decision: str                       # 'ALLOW' or 'DENY'
    rule_fired: str                     # Policy rule that made the final decision
    predicted_class: int                # 0-9 predicted traffic class
    category_name: str                  # Human-readable category
    confidence: float                   # Softmax probability for predicted class
    ml_risk_score: float                # 1 - P(Benign)
    dae_reconstruction_error: float     # Reconstruction MSE from DAE
    latency_us: float                   # Processing latency in microseconds
    latency_ms: float                   # Processing latency in milliseconds


class EdgeDefensePipeline:
    """
    Ultra-low latency edge defense engine combining ONNX runtime with Zero-Trust.
    """

    def __init__(
        self,
        classifier_onnx_path: str = ADV_ONNX_PATH,
        dae_onnx_path: Optional[str] = DAE_ONNX_PATH,
        scaler_json_path: str = SCALER_JSON_PATH,
        enable_dae_sanitization: bool = True,
        num_threads: int = 2
    ):
        self.enable_dae = enable_dae_sanitization and (dae_onnx_path is not None) and os.path.exists(dae_onnx_path)

        # 1. Configure ONNX Runtime Session Options for Edge CPU
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # 2. Load Classifier Session
        if not os.path.exists(classifier_onnx_path):
            raise FileNotFoundError(f"Classifier ONNX model not found: {classifier_onnx_path}")
        self.classifier_session = ort.InferenceSession(classifier_onnx_path, sess_options=opts, providers=['CPUExecutionProvider'])
        self.classifier_input_name = self.classifier_session.get_inputs()[0].name

        # 3. Load DAE Session (if enabled)
        if self.enable_dae:
            self.dae_session = ort.InferenceSession(dae_onnx_path, sess_options=opts, providers=['CPUExecutionProvider'])
            self.dae_input_name = self.dae_session.get_inputs()[0].name
        else:
            self.dae_session = None

        # 4. Load Scaler Parameters for lightweight in-memory preprocessing
        if os.path.exists(scaler_json_path):
            with open(scaler_json_path, 'r') as f:
                scaler_data = json.load(f)
            self.mean = np.array(scaler_data['mean'], dtype=np.float32)
            self.scale = np.array(scaler_data['scale'], dtype=np.float32)
            # Avoid divide-by-zero
            self.scale[self.scale == 0] = 1.0
        else:
            self.mean = None
            self.scale = None

        # 5. Initialize Zero-Trust Engine
        self.zt_engine = ZeroTrustEngine()

    def preprocess(self, X: np.ndarray) -> np.ndarray:
        """Standardizes raw flow features using in-memory parameters."""
        if self.mean is not None and self.scale is not None:
            return (X - self.mean) / self.scale
        return X.astype(np.float32)

    def process_flow(self, raw_flow_features: np.ndarray, context: Optional[Dict] = None) -> EdgeInferenceResult:
        """
        Processes a single network flow in real-time through the complete defense pipeline.

        Args:
            raw_flow_features: 1D numpy array of 76 flow features.
            context: Contextual signals (device_trust, geo_risk, time_of_day, identity_verified).

        Returns:
            EdgeInferenceResult dataclass with decision, scores, and latency metrics.
        """
        start_time = time.perf_counter()

        if context is None:
            context = {
                'device_trust': 0.8,
                'geo_risk': 0.2,
                'time_of_day': 12,
                'identity_verified': True,
                'resource_sensitivity': 0.5
            }

        # Step 1: Preprocess
        X = np.asarray(raw_flow_features, dtype=np.float32).reshape(1, -1)
        X_scaled = self.preprocess(X)

        # Step 2: DAE Sanitization & Anomaly Detection
        dae_recon_error = 0.0
        if self.enable_dae:
            reconstructed = self.dae_session.run(None, {self.dae_input_name: X_scaled})[0]
            dae_recon_error = float(np.mean((X_scaled - reconstructed) ** 2))
            X_input_for_classifier = reconstructed
        else:
            X_input_for_classifier = X_scaled

        # Step 3: Neural Network Classification
        logits = self.classifier_session.run(None, {self.classifier_input_name: X_input_for_classifier})[0]
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        pred_class = int(np.argmax(probs[0]))
        confidence = float(probs[0][pred_class])
        # ML Risk Score = 1 - P(Benign)
        ml_risk_score = float(1.0 - probs[0][0])

        # Step 4: Context-Aware Zero-Trust Policy Decision
        zt_decision = self.zt_engine.evaluate(ml_risk_score=ml_risk_score, context=context)

        elapsed = time.perf_counter() - start_time
        latency_ms = elapsed * 1000.0
        latency_us = elapsed * 1_000_000.0

        return EdgeInferenceResult(
            decision=zt_decision.decision,
            rule_fired=zt_decision.rule_fired,
            predicted_class=pred_class,
            category_name=ATTACK_TAXONOMY.get(pred_class, "Unknown"),
            confidence=confidence,
            ml_risk_score=ml_risk_score,
            dae_reconstruction_error=dae_recon_error,
            latency_us=latency_us,
            latency_ms=latency_ms
        )

    def process_batch(self, raw_flow_matrix: np.ndarray, contexts: Optional[List[Dict]] = None) -> List[EdgeInferenceResult]:
        """
        Batch processing optimized for vectorized edge flow stream processing.
        """
        start_time = time.perf_counter()
        n_samples = raw_flow_matrix.shape[0]

        if contexts is None:
            contexts = [{'device_trust': 0.8, 'geo_risk': 0.2, 'time_of_day': 12, 'identity_verified': True, 'resource_sensitivity': 0.5}] * n_samples

        # Vectorized Preprocessing
        X_scaled = self.preprocess(raw_flow_matrix.astype(np.float32))

        # Vectorized DAE Sanitization
        if self.enable_dae:
            reconstructed = self.dae_session.run(None, {self.dae_input_name: X_scaled})[0]
            recon_errors = np.mean((X_scaled - reconstructed) ** 2, axis=1)
            X_clf_input = reconstructed
        else:
            recon_errors = np.zeros(n_samples, dtype=np.float32)
            X_clf_input = X_scaled

        # Vectorized Classification
        logits = self.classifier_session.run(None, {self.classifier_input_name: X_clf_input})[0]
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        pred_classes = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        risk_scores = 1.0 - probs[:, 0]

        total_elapsed = time.perf_counter() - start_time
        per_flow_ms = (total_elapsed / n_samples) * 1000.0
        per_flow_us = (total_elapsed / n_samples) * 1_000_000.0

        results = []
        for i in range(n_samples):
            zt_dec = self.zt_engine.evaluate(ml_risk_score=float(risk_scores[i]), context=contexts[i])
            results.append(EdgeInferenceResult(
                decision=zt_dec.decision,
                rule_fired=zt_dec.rule_fired,
                predicted_class=int(pred_classes[i]),
                category_name=ATTACK_TAXONOMY.get(int(pred_classes[i]), "Unknown"),
                confidence=float(confidences[i]),
                ml_risk_score=float(risk_scores[i]),
                dae_reconstruction_error=float(recon_errors[i]),
                latency_us=per_flow_us,
                latency_ms=per_flow_ms
            ))

        return results


def run_edge_smoke_test():
    """Validates the edge inference pipeline on sample network flows."""
    print("=" * 75)
    print("RUNNING EDGE INFERENCE PIPELINE SMOKE TEST")
    print("=" * 75)

    pipeline = EdgeDefensePipeline(enable_dae_sanitization=True)
    sample_flow = np.zeros(76, dtype=np.float32)

    # 1. Test Clean Benign Flow
    benign_ctx = {'device_trust': 0.9, 'geo_risk': 0.1, 'time_of_day': 14, 'identity_verified': True, 'resource_sensitivity': 0.3}
    res_benign = pipeline.process_flow(sample_flow, benign_ctx)
    print(f"\n[Test 1] Benign Flow -> Decision: {res_benign.decision} | Rule: {res_benign.rule_fired} | Latency: {res_benign.latency_us:.1f} µs ({res_benign.latency_ms:.3f} ms)")

    # 2. Test Adversarial Untrusted Flow
    attack_ctx = {'device_trust': 0.1, 'geo_risk': 0.95, 'time_of_day': 2, 'identity_verified': False, 'resource_sensitivity': 0.9}
    res_attack = pipeline.process_flow(sample_flow, attack_ctx)
    print(f"[Test 2] Attacker Flow -> Decision: {res_attack.decision} | Rule: {res_attack.rule_fired} | Latency: {res_attack.latency_us:.1f} µs ({res_attack.latency_ms:.3f} ms)")

    print("\n" + "=" * 75)
    print("EDGE INFERENCE PIPELINE OPERATIONAL AND READY FOR BENCHMARKING!")
    print("=" * 75)


if __name__ == '__main__':
    run_edge_smoke_test()
