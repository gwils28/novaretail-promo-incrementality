# Bloc 4 · Backtesting : couverture, faux positifs, puissance, et ce que les placebos ne voient pas

Code : `backtest.py`. Commande : `nr backtest`. Notebook : [04](../notebooks/04_backtesting_verite_plantee.ipynb).

## Vérité plantée sur le panel réel

Des fenêtres calmes de vraies séries (aucune campagne de 150 jours avant à 21 jours après, volume ≥ 30 unités
par jour), le vrai registre de donneurs, et un effet connu écrit dans la série :

| Spécification | Ce qu'elle teste |
|---|---|
| marche 0 % | taux de faux positifs du design sur des séries jamais traitées |
| marche 5 %, 15 %, 30 % | puissance et couverture selon la taille de l'effet |
| rampe, impulsion (15 %) | formes d'intervention que la veille n'avait pas testées |
| 15 % avec *pull-forward* 50 % | le creux après la campagne, inclus dans la vérité nette |
| 15 % avec cannibalisation 50 % du premier donneur | violation de SUTVA : biais attendu positif, placebo dans le temps aveugle |

Chaque estimateur est noté sur l'effet cumulé (échelle log) : couverture de l'intervalle à 95 %, biais, RMSE,
puissance (ou faux positifs à effet nul), largeur médiane.

## Lecture

- La couverture est la note ; la largeur est un choix de modèle.
- Le taux de faux positifs mesuré à effet nul est la référence du placebo dans le temps des vraies campagnes : on
  compare la distribution du nombre de fausses dates qui se déclenchent (sur 8) à la binomiale de ce taux.
- La sensibilité au pool (taille, *leave-one-out*) est rapportée par campagne (notebook 03).
- **Recul** : le nombre de fenêtres par spécification est limité par le temps de calcul ; les couvertures portent
  une incertitude de l'ordre de ±0,1 avec 30 fenêtres. Lire les ordres de grandeur.
