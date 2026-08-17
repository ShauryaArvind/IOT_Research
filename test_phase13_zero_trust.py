from src.risk_engine.zero_trust_engine import ZeroTrustEngine


engine = ZeroTrustEngine()


print("=== PHASE 13 ZERO-TRUST TEST ===")


# --------------------------------------------------
# Test 1: Normal known traffic
# --------------------------------------------------

normal_context = {
    "device_trust": 0.9,
    "geo_risk": 0.1,
    "time_of_day": 12,
    "identity_verified": True,
    "resource_sensitivity": 0.5,
    "novelty_risk": 0.30,
}

result = engine.evaluate(
    ml_risk_score=0.20,
    context=normal_context
)

print("\nTEST 1 — Normal traffic")
print("Decision:", result.decision)
print("Rule:", result.rule_fired)
print("Novelty risk:", normal_context["novelty_risk"])


# --------------------------------------------------
# Test 2: Suspicious novelty
# --------------------------------------------------

suspicious_context = {
    "device_trust": 0.9,
    "geo_risk": 0.1,
    "time_of_day": 12,
    "identity_verified": True,
    "resource_sensitivity": 0.5,
    "novelty_risk": 0.70,
}

result = engine.evaluate(
    ml_risk_score=0.20,
    context=suspicious_context
)

print("\nTEST 2 — High novelty risk")
print("Decision:", result.decision)
print("Rule:", result.rule_fired)
print("Novelty risk:", suspicious_context["novelty_risk"])


# --------------------------------------------------
# Test 3: High ML risk
# --------------------------------------------------

ml_context = {
    "device_trust": 0.9,
    "geo_risk": 0.1,
    "time_of_day": 12,
    "identity_verified": True,
    "resource_sensitivity": 0.5,
    "novelty_risk": 0.30,
}

result = engine.evaluate(
    ml_risk_score=0.95,
    context=ml_context
)

print("\nTEST 3 — High ML risk")
print("Decision:", result.decision)
print("Rule:", result.rule_fired)


# --------------------------------------------------
# Test 4: Both ML and novelty high
# --------------------------------------------------

both_context = {
    "device_trust": 0.9,
    "geo_risk": 0.1,
    "time_of_day": 12,
    "identity_verified": True,
    "resource_sensitivity": 0.5,
    "novelty_risk": 0.70,
}

result = engine.evaluate(
    ml_risk_score=0.95,
    context=both_context
)

print("\nTEST 4 — High ML + high novelty")
print("Decision:", result.decision)
print("Rule:", result.rule_fired)


print("\n=== TEST COMPLETE ===")