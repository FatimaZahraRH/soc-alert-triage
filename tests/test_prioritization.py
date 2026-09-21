"""Tests unitaires du moteur de priorisation et de la boucle de retour d'expérience."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.triage.prioritization import estimate_attack_type_severity, triage_flow


def _make_row(**overrides) -> pd.Series:
    base = {
        "flow_id": "FLOW-000001",
        "source_asset_id": "SRC-000001",
        "asset_criticality": "medium",
        "flow_packets_per_s": 10.0,
        "bwd_packet_length_mean": 100.0,
        "flow_duration_ms": 1000.0,
        "dest_port": 443,
        "known_authorized_source": False,
    }
    base.update(overrides)
    return pd.Series(base)


class TestTriagePrioritization:
    def test_below_threshold_is_auto_closed(self):
        row = _make_row()
        decision = triage_flow(row, ml_probability=0.1, ml_threshold=0.5)
        assert decision.decision == "auto_close"

    def test_above_threshold_is_escalated(self):
        row = _make_row()
        decision = triage_flow(row, ml_probability=0.9, ml_threshold=0.5)
        assert decision.decision == "escalade"
        assert decision.priority_score > 0

    def test_known_authorized_source_is_auto_closed_despite_high_score(self):
        row = _make_row(source_asset_id="SCANNER-01")
        decision = triage_flow(row, ml_probability=0.95, ml_threshold=0.5, known_authorized_sources={"SCANNER-01"})
        assert decision.decision == "auto_close"
        assert "autorisée" in decision.reason.lower()

    def test_unknown_source_not_in_whitelist_still_escalates(self):
        row = _make_row(source_asset_id="SCANNER-01")
        decision = triage_flow(row, ml_probability=0.95, ml_threshold=0.5, known_authorized_sources=set())
        assert decision.decision == "escalade"

    def test_high_criticality_asset_increases_priority_score(self):
        row_low = _make_row(asset_criticality="low")
        row_high = _make_row(asset_criticality="high")
        decision_low = triage_flow(row_low, ml_probability=0.9, ml_threshold=0.5)
        decision_high = triage_flow(row_high, ml_probability=0.9, ml_threshold=0.5)
        assert decision_high.priority_score > decision_low.priority_score


class TestSeverityHeuristic:
    def test_high_packet_rate_flagged_as_ddos_severity(self):
        row = _make_row(flow_packets_per_s=6000)
        severity = estimate_attack_type_severity(row)
        assert severity == 3.5  # ATTACK_SEVERITY_WEIGHTS["DDoS"]

    def test_moderate_packet_rate_flagged_as_dos_severity(self):
        row = _make_row(flow_packets_per_s=600)
        severity = estimate_attack_type_severity(row)
        assert severity == 3.0  # ATTACK_SEVERITY_WEIGHTS["DoS_Hulk"]

    def test_long_duration_large_bwd_packets_flagged_as_infiltration(self):
        row = _make_row(flow_packets_per_s=10, bwd_packet_length_mean=1000, flow_duration_ms=10000)
        severity = estimate_attack_type_severity(row)
        assert severity == 4.0  # ATTACK_SEVERITY_WEIGHTS["Infiltration"]
