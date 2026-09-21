"""Tests unitaires du générateur de flux réseau synthétiques."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.flow_generator import FEATURE_COLUMNS, generate_dataset, is_malicious


class TestFlowGenerator:
    def test_generates_all_expected_classes(self):
        df = generate_dataset()
        expected = {"BENIGN", "PortScan", "WebAttack_BruteForce", "Bot", "DoS_Hulk", "DDoS", "Infiltration"}
        assert set(df["label"].unique()) == expected

    def test_all_feature_columns_present(self):
        df = generate_dataset()
        for col in FEATURE_COLUMNS:
            assert col in df.columns

    def test_no_negative_values_in_core_features(self):
        df = generate_dataset()
        for col in ["flow_duration_ms", "total_fwd_packets", "total_bwd_packets", "total_fwd_bytes", "total_bwd_bytes"]:
            assert (df[col] >= 0).all(), f"Valeurs négatives détectées dans {col}"

    def test_authorized_scan_flagged_but_labeled_benign(self):
        df = generate_dataset()
        authorized = df[df["known_authorized_source"]]
        assert len(authorized) > 0
        assert (authorized["label"] == "BENIGN").all()

    def test_authorized_sources_use_small_pool(self):
        df = generate_dataset()
        authorized = df[df["known_authorized_source"]]
        assert authorized["source_asset_id"].nunique() <= 3

    def test_non_authorized_sources_are_unique(self):
        df = generate_dataset()
        non_authorized = df[~df["known_authorized_source"]]
        assert non_authorized["source_asset_id"].nunique() == len(non_authorized)

    def test_is_malicious_helper(self):
        assert is_malicious("DDoS") is True
        assert is_malicious("BENIGN") is False

    def test_flow_ids_are_unique(self):
        df = generate_dataset()
        assert df["flow_id"].nunique() == len(df)

    def test_reproducible_with_fixed_seed(self):
        df1 = generate_dataset()
        df2 = generate_dataset()
        assert df1["label"].value_counts().to_dict() == df2["label"].value_counts().to_dict()
