"""Tests unitaires du classifieur de détection réseau."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.flow_generator import generate_dataset
from src.detection.classifier import FlowClassifier


class TestFlowClassifier:
    def test_train_returns_valid_roc_evaluation(self):
        df = generate_dataset()
        clf = FlowClassifier()
        evaluation = clf.train(df)
        assert 0.5 <= evaluation.roc_auc <= 1.0
        assert 0.0 <= evaluation.youden_threshold <= 1.0

    def test_predict_proba_bounded(self):
        df = generate_dataset()
        clf = FlowClassifier()
        clf.train(df)
        probas = clf.predict_proba(df.head(20))
        assert all(0.0 <= p <= 1.0 for p in probas)

    def test_malicious_flows_score_higher_on_average(self):
        df = generate_dataset()
        clf = FlowClassifier()
        clf.train(df)
        probas = clf.predict_proba(df)
        df = df.copy()
        df["proba"] = probas
        benign_mean = df[df["label"] == "BENIGN"]["proba"].mean()
        ddos_mean = df[df["label"] == "DDoS"]["proba"].mean()
        assert ddos_mean > benign_mean

    def test_feature_importances_not_empty_after_training(self):
        df = generate_dataset()
        clf = FlowClassifier()
        clf.train(df)
        importances = clf.feature_importances()
        assert len(importances) > 0

    def test_load_raises_if_not_trained(self, tmp_path):
        missing_path = tmp_path / "does_not_exist.joblib"
        try:
            FlowClassifier.load(missing_path)
            assert False, "Devrait lever FileNotFoundError"
        except FileNotFoundError:
            pass

    def test_predict_proba_before_training_raises(self):
        clf = FlowClassifier()
        df = generate_dataset()
        try:
            clf.predict_proba(df.head(5))
            assert False, "Devrait lever RuntimeError"
        except RuntimeError:
            pass
