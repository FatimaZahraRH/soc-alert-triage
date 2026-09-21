# SOC Alert Triage — Détection réseau & réduction de faux positifs par apprentissage

**Pipeline de détection d'intrusion réseau (inspiré CICIDS2017) + triage d'alertes contextuel + boucle de retour d'expérience analyste, inspiré de la littérature récente sur la fatigue d'alerte en SOC (ExTriage, AACT).**



---

##  Le problème traité

Un SOC réel reçoit des milliers d'alertes par jour, avec un taux de faux positifs documenté dépassant 50 % en production (Alahmadi et al., USENIX Security 2022). Un classifieur ML seul, aussi précis soit-il, ne résout pas ce problème structurellement : certains flux réseau (un scan de vulnérabilité autorisé par l'équipe sécurité, par exemple) sont **indiscernables d'une vraie attaque à partir du seul signal réseau**. Ce projet démontre ce point empiriquement, puis y répond avec une architecture à deux étages : classification réseau + triage contextuel avec apprentissage par retour d'expérience.

---

##  Architecture

```
Flux réseau (13 caractéristiques, style CICIDS2017)
      │
      ▼
┌─────────────────────┐     ┌──────────────────────────┐
│ Classifieur ML         │────▶│ Score de malveillance       │
│ (HistGradientBoosting) │     │ + seuil calibré (ROC/Youden)│
└─────────────────────┘     └──────────────────────────┘
                                          │
                                          ▼
                          ┌──────────────────────────────┐
                          │ Module de triage contextuel     │
                          │ (sévérité × criticité d'actif)  │
                          └──────────────────────────────┘
                                          │
                          ┌───────────────┴────────────────┐
                          ▼                                  ▼
              Alerte escaladée                  Source connue (retour
              à l'analyste                       d'expérience) → fermée
                                                  automatiquement
```

---

##  Structure du projet

```
soc-alert-triage/
├── src/
│   ├── config.py                     # poids de sévérité par type d'attaque
│   ├── data/
│   │   └── flow_generator.py         # générateur de flux réseau synthétiques (style CICIDS2017)
│   ├── detection/
│   │   └── classifier.py             # classifieur ML + calibration ROC/Youden
│   ├── triage/
│   │   ├── prioritization.py         # score de priorité contextuel
│   │   └── analyst_feedback.py       # boucle d'apprentissage inspirée d'AACT
│   ├── evaluation/
│   │   └── metrics.py                # comparaison avant/après apprentissage
│   ├── dashboard/
│   │   └── dashboard_builder.py      # dashboard HTML interactif (Plotly)
│   └── reporting/
│       └── report_generator.py       # rapport Markdown
├── scripts/
│   ├── generate_dataset.py
│   └── run_pipeline.py               # pipeline complet
├── tests/                            # 23 tests unitaires
└── docs/
    └── methodology_notes.md          # références de recherche et limites assumées
```

---

##  Installation et démarrage rapide

```bash
git clone <votre-repo>
cd soc-alert-triage

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Pipeline complet : génération -> entraînement -> simulation -> rapport + dashboard
python scripts/run_pipeline.py

# Tests
pytest tests/ -v
```

Aucune clé API, aucun téléchargement externe : tout tourne en local avec des données synthétiques.

---

##  Résultats mesurés

### Détection

**AUC-ROC : 0,9935** (validation croisée stratifiée, seuil calibré par indice de Youden)

### Réduction des faux positifs par apprentissage (le résultat clé)

| | Sans apprentissage | Avec apprentissage (5 jours simulés) |
|---|---|---|
| Alertes issues de sources déjà validées | 109 | **20** |
| **Réduction** | | **81,7 %** |
| Total d'alertes (tous types) | 799 | 710 |
| Attaques réelles détectées | — | 690 / 690 (100 %) |
| **Taux de faux négatifs** | | **0 %** |

La réduction des alertes ne se fait **pas** au détriment de la détection réelle — point de vigilance qui était au cœur de la méthodologie (cf. `docs/methodology_notes.md`).

### Le résultat le plus important méthodologiquement

Le classifieur réseau seul ne peut **pas** distinguer un scan de vulnérabilité autorisé d'un vrai scan malveillant : **100 % des flux d'un scanner autorisé déclenchent une alerte au premier passage**, car leur signature réseau est structurellement identique à une vraie attaque de reconnaissance. C'est précisément la raison d'être de l'architecture à deux étages de ce projet.

### Tests unitaires

23 tests couvrant : le générateur de données (classes, cohérence, reproductibilité), le classifieur (bornes, cohérence, chargement), le moteur de triage et l'heuristique de sévérité.

```bash
pytest tests/ -v
# 23 passed
```

---

##  Choix de conception à retenir

- **Séparation stricte signal réseau / contexte métier** : le classifieur ML ne voit jamais `asset_criticality` ni `known_authorized_source` — uniquement les 13 caractéristiques de flux. Le contexte est ajouté après, dans le module de triage. Cette séparation est ce qui permet de démontrer, de façon mesurable, la limite du "tout-ML" et la valeur ajoutée du triage contextuel.
- **Identifiant de source persistant** (`source_asset_id`) plutôt qu'un identifiant par flux : les scanners autorisés réutilisent un petit pool de 3 identifiants (comme de vraies machines), tandis que les autres flux ont des identifiants uniques — condition nécessaire pour qu'un apprentissage par source ait un sens réaliste, sans généraliser à tort sur de vraies attaques.
- **Permutation importance plutôt que SHAP** : choix assumé de légèreté (pas de dépendance compilée supplémentaire), documenté comme une limite par rapport à ExTriage dans `docs/methodology_notes.md`.
- **Résultat volontairement imparfait sur la réduction (81,7 %, pas 100 %)** : le jour 1 montre encore les alertes des sources autorisées avant leur première confirmation par un analyste — comportement attendu et explicable, pas un bug.

---

##  Limites assumées

- **Dataset synthétique**, pas le vrai CICIDS2017 (contrainte d'environnement) — l'AUC de 0,99 est donc optimiste par rapport à un déploiement réel sur trafic capturé, où le bruit et la diversité des comportements bénins sont plus importants.
- **Classification du type d'attaque par heuristique simple**, pas par un vrai modèle multi-classes entraîné — amélioration identifiée pour une v2.
- **Pas de vrai module RankNet appris** (contrairement à ExTriage) — le score de priorité est une pondération explicite (ML × sévérité × criticité), plus simple mais aussi plus interprétable.

Voir `docs/methodology_notes.md` pour le détail des références de recherche et de ce qui distingue ce projet de la littérature citée.

---

##  Stack technique

Python 3.12 · scikit-learn (HistGradientBoosting, permutation importance, courbe ROC) · pandas · Plotly · pytest

---

## Licence

Projet personnel à but pédagogique et de démonstration. Toutes les données réseau sont synthétiques.
