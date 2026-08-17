import json
import numpy as np
import onnxruntime as ort

from scipy.special import softmax

from sklearn.ensemble import IsolationForest



class DAENoveltyDetector:

    def __init__(
        self,
        dae_path="src/risk_engine/models/denoising_autoencoder.onnx",
        classifier_path="src/risk_engine/models/network_risk_classifier.onnx",
        scaler_path="src/risk_engine/models/scaler_params.json"
    ):
        # Load scaler parameters used by the existing Phase 10 models
        with open(scaler_path, "r") as f:
            scaler = json.load(f)

        self.mean = np.asarray(scaler["mean"], dtype=np.float32)
        self.scale = np.asarray(scaler["scale"], dtype=np.float32)

        # Load existing DAE
        self.session = ort.InferenceSession(
            dae_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Load existing Phase 10 classifier
        self.classifier_session = ort.InferenceSession(
            classifier_path,
            providers=["CPUExecutionProvider"]
        )

        self.classifier_input_name = (
            self.classifier_session.get_inputs()[0].name
        )
        

        self.classifier_output_name = (
            self.classifier_session.get_outputs()[0].name
        )
        self.isolation_forest = None

    def scale_features(self, X):
        X = np.asarray(X, dtype=np.float32)

        return (X - self.mean) / self.scale

    def reconstruction_error(self, X):
        X_scaled = self.scale_features(X)

        reconstructed = self.session.run(
            [self.output_name],
            {self.input_name: X_scaled}
        )[0]

        error = np.mean(
            (X_scaled - reconstructed) ** 2,
            axis=1
        )

        return error

    def classifier_confidence(self, X):
        X_scaled = self.scale_features(X)

        logits = self.classifier_session.run(
            [self.classifier_output_name],
            {self.classifier_input_name: X_scaled}
        )[0]

        probabilities = softmax(logits, axis=1)

        predicted_class = np.argmax(
            probabilities,
            axis=1
        )

        confidence = np.max(
            probabilities,
            axis=1
        )

        return predicted_class, confidence, probabilities

    def fit_isolation_forest(
        self,
        X,
        sample_size=100000
    ):
        X = np.asarray(X, dtype=np.float32)

        # Use a representative subset if the dataset is very large
        if len(X) > sample_size:
            rng = np.random.default_rng(42)
            indices = rng.choice(
                len(X),
                size=sample_size,
                replace=False
            )
            X = X[indices]

        X_scaled = self.scale_features(X)

        self.isolation_forest = IsolationForest(
            n_estimators=200,
            random_state=42,
            contamination="auto",
            n_jobs=-1
     )

        self.isolation_forest.fit(X_scaled)

        return self

    def isolation_anomaly_score(self, X):
        if self.isolation_forest is None:
            raise RuntimeError(
                "Isolation Forest has not been trained yet. "
                "Call fit_isolation_forest() first."
            )

        X_scaled = self.scale_features(X)

        # Higher score = more normal in sklearn.
        # Negating it makes higher values = more anomalous.
        anomaly_score = -self.isolation_forest.score_samples(
            X_scaled
        )

        return anomaly_score

    def calibrate_thresholds(self, X_known):
        """
        Learn reference thresholds from known traffic.

        The 95th percentile is used as the initial
        boundary for unusually high anomaly signals.
        """

        X_known = np.asarray(X_known, dtype=np.float32)

        # DAE reconstruction errors
        dae_errors = self.reconstruction_error(X_known)

        # Classifier confidence
        _, confidence, _ = self.classifier_confidence(X_known)

        # Isolation Forest anomaly scores
        isolation_scores = self.isolation_anomaly_score(X_known)

        self.dae_threshold = np.percentile(dae_errors, 95)

        self.confidence_threshold = np.percentile(
            confidence,
            5
        )

        self.isolation_threshold = np.percentile(
            isolation_scores,
            95
        )

        return {
            "dae_threshold": self.dae_threshold,
            "confidence_threshold": self.confidence_threshold,
            "isolation_threshold": self.isolation_threshold
        }

    def calculate_risk_scores(self, X):
        """
        Convert the three raw signals into normalized
        0–1 risk scores.
        """

        if not hasattr(self, "dae_threshold"):
            raise RuntimeError(
                "Thresholds have not been calibrated. "
                "Call calibrate_thresholds() first."
            )

        # Raw signals
        dae_errors = self.reconstruction_error(X)

        _, confidence, _ = self.classifier_confidence(X)

        isolation_scores = self.isolation_anomaly_score(X)

        # DAE risk
        dae_risk = np.clip(
            dae_errors / self.dae_threshold,
            0,
            1
        )

        # Low classifier confidence = high risk
        confidence_risk = np.clip(
            (self.confidence_threshold - confidence)
            / max(self.confidence_threshold, 1e-8),
            0,
            1
        )

        # Isolation Forest risk
        isolation_risk = np.clip(
            isolation_scores / self.isolation_threshold,
            0,
            1
        )

        return (
            dae_risk,
            confidence_risk,
            isolation_risk
        )

    def calculate_novelty_risk(self, X):
        """
        Combine DAE, classifier-confidence, and
        Isolation Forest signals into one novelty risk.
        """

        dae_risk, confidence_risk, isolation_risk = (
            self.calculate_risk_scores(X)
        )

        novelty_risk = (
            0.40 * dae_risk
            + 0.30 * confidence_risk
            + 0.30 * isolation_risk
        )

        return {
            "dae_risk": dae_risk,
            "confidence_risk": confidence_risk,
            "isolation_risk": isolation_risk,
            "novelty_risk": novelty_risk
        }