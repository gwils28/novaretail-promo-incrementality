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
