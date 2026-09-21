"""
Moteur de priorisation des alertes -- deuxième étage du pipeline, après le
classifieur réseau pur.

Principe (directement inspiré de la distinction "signal réseau" vs "contexte
métier" documentée dans la littérature SOC, ex. Alahmadi et al. 2022 sur la
fatigue d'alerte, et de l'architecture ExTriage) : un score de détection ML
seul ne suffit pas à décider quoi montrer à un analyste. Le même flux réseau
(un scan de ports) peut être une attaque réelle ou un scan de vulnérabilité
autorisé -- ce que le signal réseau seul ne peut jamais trancher.

Ce module combine :
  - la probabilité de malveillance du classifieur (src.detection.classifier)
  - la sévérité du type d'attaque (poids fixes, src.config)
  - la criticité de l'actif touché (contexte métier)
  - la connaissance d'une source autorisée connue (retour d'expérience
    analyste, cf. src/triage/analyst_feedback.py, inspiré de la logique
    AACT -- apprendre des décisions passées des analystes)
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import ATTACK_SEVERITY_WEIGHTS

_ASSET_CRITICALITY_WEIGHTS = {"low": 1.0, "medium": 1.5, "high": 2.2}


@dataclass
class TriageDecision:
    flow_id: str
    ml_probability: float
    priority_score: float
    decision: str  # "auto_close" | "escalade"
    reason: str


def estimate_attack_type_severity(row: pd.Series) -> float:
    """Heuristique simple de sévérité à partir de caractéristiques du flux
    (faute d'un vrai classifieur multi-classes dans cette v1) : combine
    volumétrie et caractère non standard du port pour approximer la
    sévérité attendue -- une amélioration future serait un vrai classifieur
    multi-classes plutôt que cette heuristique."""
    if row["flow_packets_per_s"] > 5000:
        return ATTACK_SEVERITY_WEIGHTS["DDoS"]
    if row["flow_packets_per_s"] > 500:
        return ATTACK_SEVERITY_WEIGHTS["DoS_Hulk"]
    if row["bwd_packet_length_mean"] > 800 and row["flow_duration_ms"] > 5000:
        return ATTACK_SEVERITY_WEIGHTS["Infiltration"]
    if row["dest_port"] > 10000:
        return ATTACK_SEVERITY_WEIGHTS["PortScan"]
    return ATTACK_SEVERITY_WEIGHTS["WebAttack_BruteForce"]


def triage_flow(
    row: pd.Series,
    ml_probability: float,
    ml_threshold: float,
    known_authorized_sources: set[str] | None = None,
) -> TriageDecision:
    known_authorized_sources = known_authorized_sources or set()

    if ml_probability < ml_threshold:
        return TriageDecision(
            flow_id=row["flow_id"], ml_probability=round(float(ml_probability), 4),
            priority_score=0.0, decision="auto_close", reason="Sous le seuil de détection calibré (ROC/Youden).",
        )

    # Retour d'expérience analyste (AACT-inspired) : une SOURCE déjà validée
    # comme autorisée par un analyste ne doit plus générer d'alerte, même si
    # le signal réseau seul continue de la flaguer -- et ce, même pour un
    # nouveau flux jamais vu auparavant émis par cette même source.
    if row["source_asset_id"] in known_authorized_sources:
        return TriageDecision(
            flow_id=row["flow_id"], ml_probability=round(float(ml_probability), 4),
            priority_score=0.0, decision="auto_close",
            reason="Source confirmée autorisée par un analyste (retour d'expérience).",
        )

    severity = estimate_attack_type_severity(row)
    asset_weight = _ASSET_CRITICALITY_WEIGHTS.get(row["asset_criticality"], 1.0)
    priority_score = round(float(ml_probability) * severity * asset_weight, 4)

    return TriageDecision(
        flow_id=row["flow_id"], ml_probability=round(float(ml_probability), 4),
        priority_score=priority_score, decision="escalade",
        reason=f"Score ML={ml_probability:.3f}, sévérité={severity}, actif={row['asset_criticality']}.",
    )


def triage_batch(
    df: pd.DataFrame,
    probabilities,
    ml_threshold: float,
    known_authorized_sources: set[str] | None = None,
) -> list[TriageDecision]:
    return [
        triage_flow(row, proba, ml_threshold, known_authorized_sources)
        for (_, row), proba in zip(df.iterrows(), probabilities)
    ]
