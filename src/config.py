"""Configuration centrale du projet SOC Alert Triage."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "network_flows.csv"
MODEL_PATH = DATA_DIR / "classifier_model.joblib"
DASHBOARD_OUTPUT_PATH = BASE_DIR / "dashboard.html"

RANDOM_SEED = 42

# Poids de sévérité par type d'attaque -- reprend le même principe de
# pondération contextuelle que mes projets Governance RAG Copilot et
# Mini-DSPM : toutes les alertes ne se valent pas, même à confiance de
# détection égale.
ATTACK_SEVERITY_WEIGHTS = {
    "BENIGN": 0.0,
    "PortScan": 1.5,       # reconnaissance, souvent précurseur d'attaque
    "WebAttack_BruteForce": 2.0,
    "Bot": 2.5,
    "DoS_Hulk": 3.0,
    "DDoS": 3.5,
    "Infiltration": 4.0,    # mouvement latéral / compromission active -> le plus critique
}


@dataclass
class DatasetConfig:
    n_samples_per_class: dict = field(default_factory=lambda: {
        "BENIGN": 2000,
        "PortScan": 150,
        "WebAttack_BruteForce": 100,
        "Bot": 80,
        "DoS_Hulk": 200,
        "DDoS": 120,
        "Infiltration": 40,
    })
    # Proportion des flux bénins générés comme "cas difficiles" (ressemblant
    # structurellement à une attaque) -- indispensable pour que l'évaluation
    # ROC soit significative, comme dans mes projets précédents.
    hard_negative_ratio: float = 0.12


dataset_config = DatasetConfig()
