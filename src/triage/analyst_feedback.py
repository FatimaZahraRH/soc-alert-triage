"""
Simulation d'un apprentissage par retour d'expérience analyste, inspirée de
la logique du système AACT (Turcotte et al., 2025) cité dans la revue de
littérature de ce projet : un système qui apprend des décisions passées des
analystes pour fermer automatiquement les alertes déjà validées comme
bénignes, réduisant le volume montré sans dégrader le rappel sur les
vraies attaques.

Le flux réseau seul ne peut jamais distinguer un scan de vulnérabilité
autorisé d'un vrai scan malveillant (cf. classifier.py) -- mais un analyste
qui confirme UNE fois qu'une source donnée est autorisée permet, ensuite,
de fermer automatiquement toutes les alertes futures de cette même source,
sans reperdre de temps analyste dessus.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.detection.classifier import FlowClassifier
from src.triage.prioritization import TriageDecision, triage_batch


@dataclass
class DaySimulationResult:
    day: int
    n_flows: int
    n_alerts_shown: int
    n_true_attacks_caught: int
    n_true_attacks_total: int
    n_false_negatives: int
    known_sources_count: int


def simulate_analyst_feedback_loop(
    daily_batches: list[pd.DataFrame],
    classifier: FlowClassifier,
) -> tuple[list[DaySimulationResult], list[TriageDecision]]:
    """Simule N jours de flux. Chaque jour, le pipeline classifieur + triage
    s'exécute ; les alertes dont la source est un vrai scanner autorisé
    (vérité terrain connue en interne, jamais donnée au classifieur) sont
    "confirmées" par un analyste simulé, et cette source est ajoutée à la
    liste blanche pour les jours suivants."""
    known_authorized_sources: set[str] = set()
    results: list[DaySimulationResult] = []
    all_decisions: list[TriageDecision] = []

    for day_idx, day_df in enumerate(daily_batches, start=1):
        probas = classifier.predict_proba(day_df)
        decisions = triage_batch(day_df, probas, classifier.threshold, known_authorized_sources)
        all_decisions.extend(decisions)

        escalated = [d for d in decisions if d.decision == "escalade"]

        # L'analyste simulé confirme les sources réellement autorisées parmi
        # les alertes escaladées du jour (vérité terrain, jamais vue par le
        # classifieur) et les ajoute à la liste blanche pour la suite.
        escalated_flow_ids = {d.flow_id for d in escalated}
        day_lookup = day_df.set_index("flow_id")
        for flow_id in escalated_flow_ids:
            row = day_lookup.loc[flow_id]
            if bool(row["known_authorized_source"]):
                known_authorized_sources.add(row["source_asset_id"])

        # Bilan du jour : combien de vraies attaques ont été effectivement
        # remontées à l'analyste (peu importe si elles ont aussi entraîné
        # d'autres alertes bénignes).
        day_lookup_full = day_df.set_index("flow_id")
        true_attack_ids = set(day_lookup_full[day_lookup_full["label"] != "BENIGN"].index)
        caught_ids = {d.flow_id for d in escalated} & true_attack_ids
        false_negative_ids = true_attack_ids - caught_ids

        results.append(DaySimulationResult(
            day=day_idx,
            n_flows=len(day_df),
            n_alerts_shown=len(escalated),
            n_true_attacks_caught=len(caught_ids),
            n_true_attacks_total=len(true_attack_ids),
            n_false_negatives=len(false_negative_ids),
            known_sources_count=len(known_authorized_sources),
        ))

    return results, all_decisions


def split_into_daily_batches(df: pd.DataFrame, n_days: int = 5) -> list[pd.DataFrame]:
    """Découpe le dataset en N lots chronologiques simulés. Le mélange initial
    du dataset (generate_dataset) est conservé tel quel pour la répartition,
    ce qui garantit que les scanners autorisés réapparaissent sur plusieurs
    jours -- condition nécessaire pour observer l'effet d'apprentissage."""
    return [chunk.reset_index(drop=True) for chunk in _split_dataframe(df, n_days)]


def _split_dataframe(df: pd.DataFrame, n_parts: int) -> list[pd.DataFrame]:
    chunk_size = len(df) // n_parts
    chunks = []
    for i in range(n_parts):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n_parts - 1 else len(df)
        chunks.append(df.iloc[start:end])
    return chunks
