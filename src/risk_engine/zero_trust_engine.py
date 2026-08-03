"""
Zero-Trust Policy Engine for contextual access control.

Implements prioritized policy rules that evaluate contextual signals
(device trust, geo-risk, time-of-day, identity) alongside ML risk scores
to make access decisions. Designed for defense-in-depth against adversarial
evasion: even when ML is fooled, contextual policies catch anomalous access
patterns.

Policy Architecture:
    Rules are evaluated in strict priority order (1 = highest). The first
    rule whose condition is satisfied determines the access decision. This
    mirrors real Zero-Trust Network Access (ZTNA) policy engines where
    micro-segmentation rules take precedence over broader contextual checks.

Attacker Threat Model Assumption:
    An attacker who crafts adversarial network flows is assumed to operate
    from an unregistered device (low device trust) and a geographically
    anomalous IP address (high geo-risk). This is a documented modeling
    assumption, not an empirical calibration. See context_profiles.py for
    the distributional parameters.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ==================== DATA STRUCTURES ====================

@dataclass
class PolicyRule:
    """A single Zero-Trust policy rule.

    Attributes:
        priority: Rule evaluation order (1 = highest priority).
        name: Human-readable rule identifier.
        condition_desc: Formal condition string for documentation.
        decision: Access decision when rule fires ('DENY' or 'ALLOW').
        rationale: Security rationale explaining why this rule exists.

    Example:
        >>> rule = PolicyRule(
        ...     priority=1,
        ...     name="Critical Resource Microsegment",
        ...     condition_desc="resource_sensitivity > 0.9 AND device_trust < 0.5",
        ...     decision="DENY",
        ...     rationale="High-sensitivity resources require trusted devices."
        ... )
    """
    priority: int
    name: str
    condition_desc: str
    decision: str
    rationale: str


@dataclass
class AccessDecision:
    """Result of a Zero-Trust policy evaluation.

    Attributes:
        decision: Final access decision ('ALLOW' or 'DENY').
        rule_fired: Name of the rule that determined the decision.
        priority: Priority of the rule that fired.
        ml_risk_score: The ML model's risk score for this sample.
        context: The contextual signals used in evaluation.
        all_rules_evaluated: List of (rule_name, matched) for audit trail.

    Example:
        >>> result = engine.evaluate(ml_risk=0.3, context={...})
        >>> print(f"{result.decision} by {result.rule_fired}")
        DENY by High Geo-Risk
    """
    decision: str
    rule_fired: str
    priority: int
    ml_risk_score: float
    context: Dict
    all_rules_evaluated: List[Tuple[str, bool]] = field(default_factory=list)


# ==================== POLICY ENGINE ====================

class ZeroTrustEngine:
    """Contextual Zero-Trust access control engine.

    Evaluates access requests against 8 prioritized policy rules combining
    ML risk scores with contextual signals. Supports partial context for
    ablation studies (e.g., ML-only, ML+device, ML+geo, full system).

    The 8 rules in priority order:
        1. Critical Resource Microsegment  (resource_sensitivity > 0.9 AND device_trust < 0.5)
        2. High-Risk ML Score              (ml_risk > 0.85)
        3. Device Trust Threshold          (device_trust < 0.3)
        4. Geo-Risk Threshold              (geo_risk > 0.8)
        5. Compound Risk                   (ml_risk > 0.5 AND geo_risk > 0.6)
        6. Off-Hours Anomaly               (off_hours AND device_trust < 0.5)
        7. Identity Verification Failure   (NOT identity_verified AND ml_risk > 0.4)
        8. Default Allow                   (always)

    Attributes:
        rules: List of PolicyRule objects defining the evaluation logic.
        enabled_factors: Set of context factors enabled for this instance.
            Supports: {'device_trust', 'geo_risk', 'time_of_day', 'identity'}

    Args:
        enabled_factors: Which contextual factors to use. Pass a subset
            for ablation studies. Default is all factors enabled.

    Example:
        >>> engine = ZeroTrustEngine()  # Full system
        >>> result = engine.evaluate(
        ...     ml_risk_score=0.3,
        ...     context={'device_trust': 0.2, 'geo_risk': 0.85,
        ...              'time_of_day': 14, 'identity_verified': True,
        ...              'resource_sensitivity': 0.5}
        ... )
        >>> print(result.decision, result.rule_fired)
        DENY High Geo-Risk Threshold

        >>> # Ablation: ML-only (no context)
        >>> ml_only = ZeroTrustEngine(enabled_factors=set())
        >>> result = ml_only.evaluate(ml_risk_score=0.9, context={})
        >>> print(result.decision)
        DENY
    """

    # Class-level rule definitions for documentation and paper table
    RULE_TABLE = [
        PolicyRule(
            priority=1,
            name="Critical Resource Microsegment",
            condition_desc="resource_sensitivity > 0.9 AND device_trust < 0.5",
            decision="DENY",
            rationale="High-sensitivity resources (e.g., admin panels, key stores) "
                      "require trusted devices. Micro-segmentation is evaluated first "
                      "because resource-specific boundaries are the most critical."
        ),
        PolicyRule(
            priority=2,
            name="High-Risk ML Score",
            condition_desc="ml_risk_score > 0.85",
            decision="DENY",
            rationale="When the ML classifier has very high confidence in a threat, "
                      "deny regardless of context. This catches obvious attacks that "
                      "adversarial perturbation failed to evade."
        ),
        PolicyRule(
            priority=3,
            name="Device Trust Threshold",
            condition_desc="device_trust < 0.3",
            decision="DENY",
            rationale="Unregistered or low-trust devices represent elevated risk. "
                      "Attackers crafting adversarial flows typically operate from "
                      "devices not enrolled in the organization's MDM."
        ),
        PolicyRule(
            priority=4,
            name="Geo-Risk Threshold",
            condition_desc="geo_risk > 0.8",
            decision="DENY",
            rationale="Connections from high-risk geographic regions (e.g., known "
                      "proxy/VPN exit nodes, sanctioned regions) are denied. "
                      "Adversarial operators often route through anomalous IPs."
        ),
        PolicyRule(
            priority=5,
            name="Compound Risk (ML + Geo)",
            condition_desc="ml_risk_score > 0.5 AND geo_risk > 0.6",
            decision="DENY",
            rationale="Moderate ML suspicion combined with elevated geographic risk "
                      "triggers denial. This catches adversarial samples that reduced "
                      "ML confidence below the high threshold but remain suspicious."
        ),
        PolicyRule(
            priority=6,
            name="Off-Hours Device Anomaly",
            condition_desc="time_of_day NOT IN [8,18] AND device_trust < 0.5",
            decision="DENY",
            rationale="Access from low-trust devices outside business hours represents "
                      "a behavioral anomaly consistent with after-hours intrusion attempts."
        ),
        PolicyRule(
            priority=7,
            name="Identity Verification Failure",
            condition_desc="NOT identity_verified AND ml_risk_score > 0.4",
            decision="DENY",
            rationale="Unverified identity combined with moderate ML risk suggests "
                      "potential credential compromise or session hijacking."
        ),
        PolicyRule(
            priority=8,
            name="Default Allow",
            condition_desc="TRUE (no prior rule matched)",
            decision="ALLOW",
            rationale="If no risk signal exceeds thresholds, permit access. Zero-Trust "
                      "does not mean deny-all; it means verify-then-trust."
        ),
    ]

    # Ablation configuration presets
    ABLATION_CONFIGS = {
        'ml_only': set(),
        'ml_device': {'device_trust'},
        'ml_geo': {'geo_risk'},
        'full': {'device_trust', 'geo_risk', 'time_of_day', 'identity'},
    }

    def __init__(self, enabled_factors: Optional[set] = None):
        """Initialize the Zero-Trust engine.

        Args:
            enabled_factors: Set of context factors to enable.
                Options: 'device_trust', 'geo_risk', 'time_of_day', 'identity'.
                None means all factors enabled (full system).
                Empty set means ML-only (no contextual factors).
        """
        if enabled_factors is None:
            self.enabled_factors = {'device_trust', 'geo_risk', 'time_of_day', 'identity'}
        else:
            self.enabled_factors = set(enabled_factors)

        self.rules = self.RULE_TABLE.copy()
        logger.debug(f"ZeroTrustEngine initialized with factors: {self.enabled_factors}")

    def evaluate(
        self,
        ml_risk_score: float,
        context: Dict,
    ) -> AccessDecision:
        """Evaluate an access request against the policy rule chain.

        Args:
            ml_risk_score: ML model's predicted risk score in [0, 1].
                Higher values indicate higher threat probability.
            context: Contextual signals dictionary. Keys:
                - device_trust (float, 0-1): Device enrollment/trust level.
                - geo_risk (float, 0-1): Geographic anomaly score.
                - time_of_day (int, 0-23): Hour of access attempt.
                - identity_verified (bool): Whether identity was verified.
                - resource_sensitivity (float, 0-1): Target resource sensitivity.

        Returns:
            AccessDecision with the final decision, firing rule, and audit trail.

        Example:
            >>> engine = ZeroTrustEngine()
            >>> decision = engine.evaluate(0.3, {
            ...     'device_trust': 0.8, 'geo_risk': 0.2,
            ...     'time_of_day': 10, 'identity_verified': True,
            ...     'resource_sensitivity': 0.5
            ... })
            >>> decision.decision
            'ALLOW'
        """
        # Extract context with safe defaults
        device_trust = context.get('device_trust', 0.8)
        geo_risk = context.get('geo_risk', 0.2)
        time_of_day = context.get('time_of_day', 12)
        identity_verified = context.get('identity_verified', True)
        resource_sensitivity = context.get('resource_sensitivity', 0.5)

        is_off_hours = time_of_day < 8 or time_of_day > 18

        # Define rule conditions — disabled factors always evaluate to False (non-triggering)
        rule_conditions = [
            # Rule 1: Critical Resource Microsegment
            (
                'device_trust' in self.enabled_factors
                and resource_sensitivity > 0.9
                and device_trust < 0.5
            ),
            # Rule 2: High-Risk ML Score (always active — ML is the base layer)
            ml_risk_score > 0.85,
            # Rule 3: Device Trust Threshold
            (
                'device_trust' in self.enabled_factors
                and device_trust < 0.3
            ),
            # Rule 4: Geo-Risk Threshold
            (
                'geo_risk' in self.enabled_factors
                and geo_risk > 0.8
            ),
            # Rule 5: Compound Risk (ML + Geo)
            (
                'geo_risk' in self.enabled_factors
                and ml_risk_score > 0.5
                and geo_risk > 0.6
            ),
            # Rule 6: Off-Hours Device Anomaly
            (
                'time_of_day' in self.enabled_factors
                and 'device_trust' in self.enabled_factors
                and is_off_hours
                and device_trust < 0.5
            ),
            # Rule 7: Identity Verification Failure
            (
                'identity' in self.enabled_factors
                and not identity_verified
                and ml_risk_score > 0.4
            ),
            # Rule 8: Default Allow (always True)
            True,
        ]

        # Evaluate rules in priority order
        audit_trail = []
        for i, (rule, condition) in enumerate(zip(self.rules, rule_conditions)):
            matched = bool(condition)
            audit_trail.append((rule.name, matched))

            if matched:
                return AccessDecision(
                    decision=rule.decision,
                    rule_fired=rule.name,
                    priority=rule.priority,
                    ml_risk_score=ml_risk_score,
                    context=context.copy(),
                    all_rules_evaluated=audit_trail,
                )

        # Should never reach here due to default allow, but safety fallback
        return AccessDecision(
            decision="ALLOW",
            rule_fired="Fallback",
            priority=99,
            ml_risk_score=ml_risk_score,
            context=context.copy(),
            all_rules_evaluated=audit_trail,
        )

    def evaluate_batch(
        self,
        ml_risk_scores: np.ndarray,
        contexts: List[Dict],
    ) -> List[AccessDecision]:
        """Evaluate a batch of access requests.

        Args:
            ml_risk_scores: Array of ML risk scores, shape (n_samples,).
            contexts: List of context dicts, one per sample.

        Returns:
            List of AccessDecision objects.

        Example:
            >>> scores = np.array([0.9, 0.3, 0.6])
            >>> ctxs = [{'device_trust': 0.2}, {'device_trust': 0.9}, {'device_trust': 0.4}]
            >>> decisions = engine.evaluate_batch(scores, ctxs)
            >>> [d.decision for d in decisions]
            ['DENY', 'ALLOW', 'DENY']
        """
        results = []
        for score, ctx in zip(ml_risk_scores, contexts):
            results.append(self.evaluate(float(score), ctx))
        return results

    def get_rule_table_for_paper(self) -> List[Dict]:
        """Export the rule table in a format suitable for paper inclusion.

        Returns:
            List of dicts with keys: priority, name, condition, decision, rationale.

        Example:
            >>> table = engine.get_rule_table_for_paper()
            >>> for row in table:
            ...     print(f"Rule {row['priority']}: {row['name']}")
        """
        return [
            {
                'priority': r.priority,
                'name': r.name,
                'condition': r.condition_desc,
                'decision': r.decision,
                'rationale': r.rationale,
            }
            for r in self.rules
        ]


# ==================== CONVENIENCE FUNCTION ====================

def evaluate_access(
    ml_risk_score: float,
    context: Dict,
    enabled_factors: Optional[set] = None,
) -> AccessDecision:
    """Convenience function for single-shot access evaluation.

    Creates a ZeroTrustEngine with the specified factors and evaluates
    a single access request. For batch evaluation, instantiate the
    engine directly.

    Args:
        ml_risk_score: ML model's predicted risk score in [0, 1].
        context: Contextual signals dictionary.
        enabled_factors: Which context factors to enable (None = all).

    Returns:
        AccessDecision with decision, rule name, and audit trail.

    Example:
        >>> result = evaluate_access(0.3, {'device_trust': 0.2, 'geo_risk': 0.9})
        >>> print(result.decision)
        DENY
    """
    engine = ZeroTrustEngine(enabled_factors=enabled_factors)
    return engine.evaluate(ml_risk_score, context)