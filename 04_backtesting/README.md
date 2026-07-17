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

## Résultats (fichiers Kaggle, `nr backtest --n 30`, `data/results/planted_scores.parquet`)

Effet cumulé sur l'échelle log, 28 à 30 fenêtres par spécification, six estimateurs.

| Spécification | `bsts_llevel` couverture / puissance | `augmented_sc` | `ols` = `bsts_static` | `synthetic_control` | `diff_in_diff` |
|---|---|---|---|---|---|
| marche 0 % (faux positifs) | 0,83 / **0,17** | 0,62 / 0,38 | 0,59 / 0,41 | 0,59 / 0,41 | 0,66 / 0,34 |
| marche 5 % | 0,83 / 0,20 | 0,63 / 0,43 | 0,63 / 0,37 | 0,60 / 0,47 | 0,73 / 0,20 |
| marche 15 % | 0,83 / 0,40 | 0,90 / 0,23 | 0,60 / 0,60 | 0,80 / 0,27 | 0,73 / 0,30 |
| marche 30 % | 0,89 / 0,21 | 0,68 / 0,54 | 0,68 / 0,61 | 0,68 / 0,57 | 0,68 / 0,61 |
| rampe 15 % | 0,90 / 0,10 | 0,67 / 0,33 | 0,67 / 0,40 | 0,67 / 0,37 | 0,70 / 0,27 |
| impulsion 15 % | 0,83 / 0,17 | 0,62 / 0,45 | 0,52 / 0,48 | 0,55 / 0,41 | 0,69 / 0,28 |
| 15 % + pull-forward 50 % | 0,90 / 0,17 | 0,67 / 0,37 | 0,53 à 0,60 / 0,47 | 0,63 / 0,43 | 0,70 / 0,33 |
| 15 % + cannibalisation 50 % | 0,86 / 0,31 | 0,72 / 0,41 | 0,66 / 0,52 | 0,66 / 0,45 | 0,79 / 0,14 |

Largeur médiane des intervalles (marche 15 %) : `bsts_llevel` 4,7 ; `augmented_sc` 3,9 ; `ols` 2,9 ; `synthetic_control` 4,0 ; `diff_in_diff` 5,6.

**Lecture.**
- Sur les vraies séries, **personne n'atteint 95 %**. Le niveau local est le moins mauvais (0,83 à 0,90) ; les
  estimateurs à erreurs iid sont à 0,5 à 0,7 et se déclenchent sur du vide 4 fois sur 10. La veille avait mesuré
  0,92 à 0,94 pour le niveau local sur panels synthétiques : les séries réelles (zéros, valeurs extrêmes, ruptures
  de régime) coûtent une dizaine de points de couverture.
- Le **taux de faux positifs de 17 %** à effet nul est la référence pour lire les placebos des vraies campagnes :
  avec huit fausses dates, on attend 1,4 déclenchement en moyenne et plus de 2 dans 23 % des cas sur du vide.
  Le seuil du protocole (au plus 2 sur 8) rejette donc un quart des campagnes mesurables par construction.
- La puissance est faible : un lift de 15 % sur sept jours est détecté 4 fois sur 10 ; de 5 %, 2 fois sur 10,
  soit à peine au-dessus du taux de faux positifs. **Ce design ne sait pas mesurer une petite promotion sur une
  seule série magasin-famille** ; il faut agréger (famille × cluster) ou accepter de ne conclure que sur les
  grands effets.
- La cannibalisation d'un donneur ne biaise presque pas le niveau local ici (+0,17 en log cumulé) : le registre
  plafonne la contribution d'un donneur parmi douze. Les estimateurs à poids concentrés (SC) y sont plus sensibles.
- **Recul** : 30 fenêtres par cellule, soit ±0,1 sur chaque couverture ; les RMSE de 10 à 25 de certains
  estimateurs viennent de quelques fenêtres où un donneur saute (changement de régime d'enregistrement) et où un
  modèle sans niveau stochastique extrapole ce saut.
