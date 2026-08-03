"""
Network Risk Classifier (Multiclass)
ML-based intrusion detection model for network traffic.

Adapted from the original binary NetworkRiskClassifier to output
class logits over `num_classes` attack categories (0 = benign,
1-9 = attack types) instead of a single sigmoid risk score.
"""

import torch
import torch.nn as nn


class NetworkRiskClassifier(nn.Module):
    """
    Neural network for multiclass network intrusion detection.
    Outputs raw logits over `num_classes` categories.
    """

    def __init__(self, input_dim=9, num_classes=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),  # raw logits, no softmax here —
                                          # CrossEntropyLoss applies it internally
        )

    def forward(self, x):
        """Forward pass through network. Returns raw logits (N, num_classes)."""
        return self.network(x)

    def predict_batch(self, X):
        """
        Predict class labels for a batch of network flows.

        Args:
            X: Batch of network flow features (numpy array or tensor)

        Returns:
            Predicted class indices as a numpy array
        """
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)

        self.eval()
        with torch.no_grad():
            logits = self.forward(X)
            preds = torch.argmax(logits, dim=1)
            return preds.numpy()

    def predict_proba_batch(self, X):
        """
        Predict per-class probabilities for a batch of network flows.

        Args:
            X: Batch of network flow features (numpy array or tensor)

        Returns:
            Softmax probabilities as a numpy array, shape (N, num_classes)
        """
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)

        self.eval()
        with torch.no_grad():
            logits = self.forward(X)
            probs = torch.softmax(logits, dim=1)
            return probs.numpy()