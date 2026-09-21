# Notes méthodologiques et références

Ce projet s'inspire de plusieurs travaux de recherche récents sur le triage d'alertes SOC et la réduction de la fatigue d'alerte. Cette note résume, dans mes propres mots, ce que j'en ai retenu et comment cela a influencé les choix de conception.

## 1. Le problème de fond : la fatigue d'alerte

Alahmadi, Axon et Martinovic (USENIX Security 2022, *"99% False Positives: A Qualitative Study of SOC Analysts' Perspectives on Security Alarms"*) documentent, à partir d'entretiens avec des analystes SOC réels, que le taux de faux positifs perçu dans les environnements de production dépasse très largement les 50 %, au point de provoquer un désengagement des analystes vis-à-vis des alertes. C'est ce constat qui motive la conception à deux étages de ce projet : un classifieur réseau seul, aussi précis soit-il, ne suffit pas -- il faut aussi un mécanisme de priorisation contextuelle qui apprend dans le temps.

## 2. AACT -- apprendre des décisions des analystes

Turcotte et al. (2025), cités dans une synthèse de 2026 sur le triage d'alertes assisté par IA, décrivent un système (AACT) qui imite les actions passées des analystes pour fermer automatiquement les alertes bénignes et escalader les alertes critiques. Le déploiement réel documenté rapporte une réduction de 61 % des alertes montrées aux analystes, pour un taux de faux négatifs de 1,36 %.

Le module `src/triage/analyst_feedback.py` de ce projet reprend ce principe : une source une fois confirmée bénigne par un "analyste" (simulé) est automatiquement exclue des alertes futures. Mes propres résultats (81,7 % de réduction des alertes issues de sources déjà validées, 0 % de faux négatif sur le jeu de test) sont du même ordre de grandeur, ce qui est cohérent avec la littérature -- même si mon jeu de données est synthétique et donc structurellement plus favorable qu'un environnement réel.

## 3. ExTriage -- classification explicable et priorisation

ExTriage combine un ensemble XGBoost + MLP, une couche d'explicabilité (SHAP, explications contrefactuelles) et un module de priorisation inspiré de RankNet. Évalué sur CICIDS-2017, CICIDS-2018 et UNSW-NB15, il atteint un F1 de 97,89 % et une réduction de 70,8 % des faux positifs.

Ce projet en reprend l'esprit général (classification + explicabilité + priorisation) avec des choix plus légers, assumés et documentés :
- **HistGradientBoostingClassifier (scikit-learn)** plutôt que XGBoost, pour éviter une dépendance compilée supplémentaire.
- **Permutation importance** plutôt que SHAP, pour la même raison de légèreté -- une vraie limite par rapport à ExTriage (SHAP donne des explications par instance, la permutation importance reste globale).
- **Score de priorité pondéré (ML × sévérité × criticité d'actif)** plutôt qu'un vrai modèle RankNet appris -- une simplification à assumer clairement en entretien.

## 4. Dataset

La structure du dataset (caractéristiques de flux, catégories d'attaques) s'inspire de CICIDS2017 (Sharafaldin, Lashkari et Ghorbani, 2018), un jeu de données largement utilisé dans la littérature citée ci-dessus. Le dataset réel n'a pas été téléchargé dans cet environnement (contrainte réseau + volumétrie) ; un générateur synthétique reproduit des signatures statistiques réalistes par type d'attaque, avec des cas volontairement ambigus (trafic bénin ressemblant structurellement à une attaque) pour que l'évaluation ROC reste significative.

## Limites assumées

- Le dataset est **synthétique**, pas le vrai CICIDS2017 -- les résultats (AUC 0,99) sont donc optimistes par rapport à un déploiement réel sur trafic capturé.
- Pas de vrai SHAP, pas de vrai RankNet appris -- des substituts plus légers mais moins puissants.
- La classification multi-classes du type d'attaque est une heuristique simple (`estimate_attack_type_severity`), pas un vrai classifieur entraîné -- une amélioration identifiée mais non traitée dans cette v1.
