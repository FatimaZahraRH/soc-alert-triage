#!/usr/bin/env python3
"""
Pipeline complet : génération du dataset -> entraînement + calibration ROC ->
simulation de la boucle de retour d'expérience analyste -> rapport + dashboard.

Usage :
    python scripts/run_pipeline.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DASHBOARD_OUTPUT_PATH
from src.data.flow_generator import generate_dataset
from src.dashboard.dashboard_builder import build_dashboard_html
from src.detection.classifier import FlowClassifier
from src.evaluation.metrics import evaluate_feedback_loop
from src.reporting.report_generator import generate_report, save_report
from src.triage.analyst_feedback import split_into_daily_batches

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Étape 1/4 -- Génération du dataset synthétique...")
    df = generate_dataset()
    logger.info("%d flux générés (%d classes).", len(df), df["label"].nunique())

    logger.info("Étape 2/4 -- Entraînement du classifieur + calibration ROC/Youden...")
    classifier = FlowClassifier()
    roc_evaluation = classifier.train(df)
    logger.info("AUC=%.4f, seuil calibré=%.4f", roc_evaluation.roc_auc, roc_evaluation.youden_threshold)

    logger.info("Étape 3/4 -- Simulation de la boucle de retour d'expérience analyste (5 jours)...")
    daily_batches = split_into_daily_batches(df, n_days=5)
    comparison, day_results = evaluate_feedback_loop(daily_batches, classifier, df)
    logger.info(
        "Réduction des alertes de sources autorisées : %d -> %d (%.1f %%)",
        comparison.baseline_alerts_from_authorized, comparison.learned_alerts_from_authorized, comparison.reduction_pct,
    )
    logger.info(
        "Taux de faux négatifs : %.2f %% (%d/%d attaques captées)",
        comparison.false_negative_rate_learned * 100,
        comparison.total_true_attacks_caught_learned, comparison.total_true_attacks,
    )

    logger.info("Étape 4/4 -- Génération du rapport et du dashboard...")
    feature_importances = classifier.feature_importances()
    report_md = generate_report(roc_evaluation, comparison, day_results, feature_importances)
    report_path = Path(__file__).resolve().parent.parent / "rapport_soc.md"
    save_report(report_md, str(report_path))
    logger.info("Rapport sauvegardé : %s", report_path)

    build_dashboard_html(roc_evaluation, comparison, day_results, feature_importances, str(DASHBOARD_OUTPUT_PATH))
    logger.info("Dashboard sauvegardé : %s", DASHBOARD_OUTPUT_PATH)

    print("\n" + "=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)
    print(f"AUC-ROC : {roc_evaluation.roc_auc}")
    print(f"Réduction alertes (sources autorisées) : {comparison.reduction_pct} %")
    print(f"Taux de faux négatifs : {comparison.false_negative_rate_learned*100:.2f} %")
    print(f"Total alertes : {comparison.total_alerts_baseline} (sans apprentissage) "
          f"-> {comparison.total_alerts_learned} (avec apprentissage)")


if __name__ == "__main__":
    main()
