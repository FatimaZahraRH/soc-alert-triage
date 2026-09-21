"""
Classifieur de détection réseau : distingue trafic bénin et malveillant à
partir des seules caractéristiques de flux (aucun contexte métier ici --
le contexte est ajouté ensuite par le module de triage).

Méthodologie identique aux 3 autres projets du portfolio (Cassiopée,
RAG Red-Team, Mini-DSPM) : validation croisée stratifiée + calibration du
seuil de décision par courbe ROC et indice de Youden, plutôt qu'un seuil
arbitraire.

Choix d'implémentation : HistGradientBoostingClassifier (scikit-learn) plutôt
que XGBoost, pour éviter une dépendance compilée supplémentaire -- même
famille d'algorithme (boosting d'arbres), architecture assez proche de celle
utilisée par ExTriage (XGBoost + MLP) pour rester comparable en pratique.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MODEL_PATH, RANDOM_SEED
from src.data.flow_generator import FEATURE_COLUMNS, is_malicious

logger = logging.getLogger(__name__)


@dataclass
class ROCEvaluation:
    roc_auc: float
    youden_threshold: float
    youden_j: float
    fpr: list = field(default_factory=list)
    tpr: list = field(default_factory=list)


class FlowClassifier:
    def __init__(self):
        self._model = None
        self.threshold: float = 0.5
        self._feature_importances: dict[str, float] = {}

    def train(self, df: pd.DataFrame) -> ROCEvaluation:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.inspection import permutation_importance
        from sklearn.metrics import roc_auc_score, roc_curve
        from sklearn.model_selection import StratifiedKFold, cross_val_predict

        X = df[FEATURE_COLUMNS].values
        y = df["label"].apply(is_malicious).astype(int).values

        self._model = HistGradientBoostingClassifier(random_state=RANDOM_SEED, max_iter=200)

        # Évaluation honnête par validation croisée stratifiée (chaque
        # prédiction utilisée pour l'évaluation vient d'un modèle qui n'a
        # PAS vu cet exemple pendant son propre entraînement).
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        cross_val_probas = cross_val_predict(self._model, X, y, cv=cv, method="predict_proba")[:, 1]

        fpr, tpr, thresholds = roc_curve(y, cross_val_probas)
        auc = roc_auc_score(y, cross_val_probas)
        youden_j_values = tpr - fpr
        best_idx = int(youden_j_values.argmax())
        youden_threshold = float(thresholds[best_idx])

        self.threshold = youden_threshold

        # Entraînement final sur 100% des données pour le modèle livré.
        self._model.fit(X, y)

        # Importance des caractéristiques par permutation (substitut à SHAP,
        # cf. docstring du module) -- calculée une fois à l'entraînement.
        perm_result = permutation_importance(
            self._model, X, y, n_repeats=8, random_state=RANDOM_SEED, scoring="roc_auc",
        )
        self._feature_importances = {
            col: round(float(importance), 4)
            for col, importance in zip(FEATURE_COLUMNS, perm_result.importances_mean)
        }

        self._save()

        evaluation = ROCEvaluation(
            roc_auc=round(float(auc), 4),
            youden_threshold=round(youden_threshold, 4),
            youden_j=round(float(youden_j_values[best_idx]), 4),
            fpr=[round(float(x), 4) for x in fpr],
            tpr=[round(float(x), 4) for x in tpr],
        )
        logger.info("Classifieur entraîné : AUC=%.4f, seuil Youden=%.4f", evaluation.roc_auc, evaluation.youden_threshold)
        return evaluation

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Modèle non entraîné/chargé.")
        return self._model.predict_proba(df[FEATURE_COLUMNS].values)[:, 1]

    def feature_importances(self) -> dict[str, float]:
        """Importance globale des caractéristiques (permutation importance,
        calculée à l'entraînement), utilisée pour l'explicabilité des
        alertes -- substitut à SHAP (cf. src/detection/explainability.py)."""
        return dict(self._feature_importances)

    def _save(self, path: Path = MODEL_PATH) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self._model, "threshold": self.threshold, "feature_importances": self._feature_importances},
            path,
        )

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "FlowClassifier":
        import joblib

        if not path.exists():
            raise FileNotFoundError(f"Aucun modèle entraîné trouvé à {path}. Lancez scripts/train_and_evaluate.py.")
        state = joblib.load(path)
        instance = cls()
        instance._model = state["model"]
        instance.threshold = state["threshold"]
        instance._feature_importances = state.get("feature_importances", {})
        return instance
