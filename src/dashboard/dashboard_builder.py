"""Dashboard HTML statique (Plotly), auto-suffisant, présentant le bilan du
pipeline de détection et de triage."""
from __future__ import annotations

from src.detection.classifier import ROCEvaluation
from src.evaluation.metrics import FeedbackLoopComparison
from src.triage.analyst_feedback import DaySimulationResult


def build_dashboard_html(
    roc_evaluation: ROCEvaluation,
    comparison: FeedbackLoopComparison,
    day_results: list[DaySimulationResult],
    feature_importances: dict,
    output_path: str,
) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scatter"}, {"type": "bar"}], [{"type": "bar"}, {"type": "bar"}]],
        subplot_titles=(
            "Courbe ROC du classifieur réseau",
            "Alertes issues de sources autorisées : avant / après apprentissage",
            "Volume d'alertes montrées par jour",
            "Importance des caractéristiques (permutation)",
        ),
        vertical_spacing=0.16,
    )

    # --- ROC ---
    fig.add_trace(
        go.Scatter(x=roc_evaluation.fpr, y=roc_evaluation.tpr, mode="lines",
                    name=f"ROC (AUC={roc_evaluation.roc_auc})", line=dict(color="#1565C0", width=3)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Aléatoire", line=dict(color="#BDBDBD", dash="dash")),
        row=1, col=1,
    )

    # --- Réduction alertes sources autorisées ---
    fig.add_trace(
        go.Bar(
            x=["Sans apprentissage", "Avec apprentissage"],
            y=[comparison.baseline_alerts_from_authorized, comparison.learned_alerts_from_authorized],
            marker_color=["#E65100", "#2E7D32"],
            text=[comparison.baseline_alerts_from_authorized, comparison.learned_alerts_from_authorized],
            textposition="outside",
        ),
        row=1, col=2,
    )

    # --- Alertes par jour ---
    days = [r.day for r in day_results]
    alerts_shown = [r.n_alerts_shown for r in day_results]
    known_sources = [r.known_sources_count for r in day_results]
    fig.add_trace(
        go.Bar(x=[f"Jour {d}" for d in days], y=alerts_shown, marker_color="#5E35B1", name="Alertes montrées"),
        row=2, col=1,
    )

    # --- Feature importances ---
    sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:8]
    fig.add_trace(
        go.Bar(x=[f[1] for f in sorted_features], y=[f[0] for f in sorted_features], orientation="h",
               marker_color="#00838F"),
        row=2, col=2,
    )

    fig.update_xaxes(title_text="Taux de faux positifs (FPR)", row=1, col=1)
    fig.update_yaxes(title_text="Taux de vrais positifs (TPR)", row=1, col=1)
    fig.update_yaxes(title_text="Nombre d'alertes", row=1, col=2)
    fig.update_yaxes(title_text="Alertes montrées", row=2, col=1)
    fig.update_xaxes(title_text="Importance (permutation, AUC)", row=2, col=2)

    fig.update_layout(
        title_text="SOC Alert Triage -- Détection réseau & réduction de faux positifs par apprentissage",
        height=800, showlegend=False, template="plotly_white",
    )

    fig.write_html(output_path, include_plotlyjs=True, full_html=True)
