# Bloc 6 · Monitoring : surveiller ce qu'on ne peut pas observer

Code : `monitoring.py`, `calendar_opt.py`. Commandes : `nr monitor`, `nr calendar`.
Notebook : [07](../notebooks/07_monitoring_calendrier.ipynb).

## Quatre signaux, un garde-fou

| Signal | Garde | Règle | Mode de défaillance reproduit au bloc 4 |
|---|---|---|---|
| 1. Ajustement de pré-période par campagne | le modèle est de la bonne taille | R² < 0,5, couverture un pas hors [0,90 ; 0,98] ou Ljung-Box p < 0,01 bloque le rapport | série trop bruitée, saisonnalité non captée |
| 2. Placebo roulant sur fenêtres jamais traitées | calibration des intervalles | taux de faux positifs sur les 50 derniers runs > 10 % | pré-période courte, intervalles trop étroits |
| 3. Isolation des donneurs par campagne | interférence | plus d'un donneur retiré → revue | cannibalisation d'un donneur |
| 4. Stabilité donneur-cible entre campagnes | la relation extrapolée | changement de signe d'un coefficient → donneur bloqué pour cette cible | dérive des chargements |
| **Réconciliation de marge** | ce que rien d'autre ne voit | écart réalisé − impliqué du même signe trois mois de suite au-delà de 2 % | rupture de relation à la date de campagne |

La décision prend la **borne basse** de l'intervalle tant que la famille n'est pas réconciliée.

## Ce qui est simulé

Favorita n'a pas de marge. Le grand livre est simulé à partir des estimations (bruit proportionnel à l'intervalle,
dérive de 2 % par mois injectée à une date connue) pour montrer que l'alarme se déclenche sur la dérive et pas sur
le bruit. Le code est celui qu'un vrai grand livre alimenterait.

## Le calendrier

Réallocation gloutonne des jours de campagne par magasin, à budget constant, vers la meilleure valeur par jour
(uplift net, borne basse, par jour de campagne mesuré), plafond de deux fois l'historique par famille, familles
non mesurées inchangées, familles liées par une arête de substitution jamais la même semaine. Ce qui bouge est
petit : la plupart des familles n'ont pas de campagne mesurable, et c'est l'information la plus utile du projet
pour la direction de NovaRetail.

## Résultats (fichiers Kaggle, `nr monitor`, `nr calendar`)

| Signal | Valeur sur les 284 campagnes estimées (pool suffisant) |
|---|---|
| 1. Bloquées par l'ajustement de pré-période | 71 (R² < 0,5 : 55 ; couverture un pas hors bornes : 21 ; Ljung-Box : 5) |
| 2. Faux positifs sur fenêtres jamais traitées | **17 %** (nominal 5 %) ; le taux roulant dépasse 10 % sur deux tiers de la fenêtre : le design est en alerte permanente, ses intervalles ne couvrent pas |
| 3. Campagnes où l'isolation retire plus d'un donneur | 114 : avec 17 % de faux positifs par donneur et douze donneurs, l'isolation en retire deux par bruit ; le signal 3 n'est lisible qu'une fois le signal 2 revenu sous 10 % |
| 4. Changements de signe d'un coefficient donneur entre deux campagnes d'une même cible | 223 : les coefficients d'une régression sur douze donneurs corrélés ne sont pas stables un par un ; ce signal doit porter sur la prévision du pool, pas sur chaque coefficient |
| Réconciliation, grand livre propre | 1 fausse alarme sur 44 mois |
| Réconciliation, dérive de 2 %/mois injectée en avril 2016 | **11 alarmes après l'injection, la première en juillet 2016** (trois mois de dérive, comme la règle le demande) |

| Calendrier | Valeur |
|---|---|
| Budget (jours de campagne historiques des 301 campagnes estimées) | 2 327 jours sur 53 magasins |
| Jours déplacés | 48 (2 %) |
| Jours non dépensés | **1 500 (64 %)** : aucune famille mesurée du magasin n'a de valeur par jour positive sur la borne basse |
| Familles jamais mesurables | 10 sur 33 |

**Lecture.** Le calendrier dit ce que l'estimation dit : sur la borne basse, presque aucune campagne ne vaut ses
jours. Deux lectures coexistent et le projet ne tranche pas entre elles : les promotions de NovaRetail sont pour
l'essentiel non incrémentales (le soupçon de la direction), ou le design n'a pas la puissance de voir des effets
de 5 à 15 % sur une série magasin-famille (bloc 4 : 20 à 40 %). La sortie utile n'est pas le plan de réallocation,
c'est la liste des campagnes et familles où la mesure est possible, et la recommandation d'un **holdout réel**
(magasins témoins tirés au sort) sur quelques campagnes pour ancrer le contrefactuel, comme la veille le proposait.
