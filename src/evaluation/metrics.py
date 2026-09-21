"""Métriques d'évaluation du pipeline complet : réduction d'alertes grâce à
la boucle de retour d'expérience analyste, et bilan global de détection."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.detection.classifier import FlowClassifier
from src.triage.analyst_feedback import DaySimulationResult, simulate_analyst_feedback_loop
from src.triage.prioritization import triage_batch


@dataclass
class FeedbackLoopComparison:
    baseline_alerts_from_authorized: int
    learned_alerts_from_authorized: int
    reduction_pct: float
    total_alerts_baseline: int
    total_alerts_learned: int
    total_true_attacks: int
    total_true_attacks_caught_learned: int
    false_negative_rate_learned: float


def evaluate_feedback_loop(daily_batches: list[pd.DataFrame], classifier: FlowClassifier, full_df: pd.DataFrame) -> tuple[FeedbackLoopComparison, list[DaySimulationResult]]:
    lookup = full_df.set_index("flow_id")

    # Scénario "avec apprentissage" (boucle normale)
    day_results, decisions = simulate_analyst_feedback_loop(daily_batches, classifier)

    learned_alerts_from_authorized = sum(
        1 for d in decisions
        if d.decision == "escalade" and bool(lookup.loc[d.flow_id, "known_authorized_source"])
    )
    total_alerts_learned = sum(1 for d in decisions if d.decision == "escalade")

    # Scénario "sans apprentissage" (baseline : liste blanche toujours vide)
    baseline_alerts_from_authorized = 0
    total_alerts_baseline = 0
    for day_df in daily_batches:
        probas = classifier.predict_proba(day_df)
        baseline_decisions = triage_batch(day_df, probas, classifier.threshold, known_authorized_sources=set())
        for d in baseline_decisions:
            if d.decision == "escalade":
                total_alerts_baseline += 1
                if bool(lookup.loc[d.flow_id, "known_authorized_source"]):
                    baseline_alerts_from_authorized += 1

    reduction_pct = 0.0
    if baseline_alerts_from_authorized > 0:
        reduction_pct = round(
            (baseline_alerts_from_authorized - learned_alerts_from_authorized) / baseline_alerts_from_authorized * 100, 1,
        )

    total_true_attacks = sum(r.n_true_attacks_total for r in day_results)
    total_caught = sum(r.n_true_attacks_caught for r in day_results)
    total_fn = sum(r.n_false_negatives for r in day_results)
    fn_rate = round(total_fn / total_true_attacks, 4) if total_true_attacks > 0 else 0.0

    comparison = FeedbackLoopComparison(
        baseline_alerts_from_authorized=baseline_alerts_from_authorized,
        learned_alerts_from_authorized=learned_alerts_from_authorized,
        reduction_pct=reduction_pct,
        total_alerts_baseline=total_alerts_baseline,
        total_alerts_learned=total_alerts_learned,
        total_true_attacks=total_true_attacks,
        total_true_attacks_caught_learned=total_caught,
        false_negative_rate_learned=fn_rate,
    )
    return comparison, day_results
