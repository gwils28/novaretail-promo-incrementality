# Bloc 1 · Données : le panel, ses zéros, son calendrier, son traitement

Code : `src/novaretail/data.py`, `calendar.py`, `features.py`, `campaigns.py`, `donors.py`, `dgp.py`.
Commandes : `nr ingest`, `nr campaigns`. Notebooks : [01](../notebooks/01_donnees_panel_zeros.ipynb),
[02](../notebooks/02_identification_campagnes_donneurs.ipynb).

## Ce que le bloc décide

| Question du cahier des charges | Décision | Pourquoi |
|---|---|---|
| « Gérer l'absence de lignes à ventes nulles » | Le fichier *Store Sales* (2021) est dense ; on vérifie la densité (`rows_expected == rows_file + lignes du 25 décembre`) et on **distingue les zéros structurels** (`before_opening`, `never_sold`, `closed`) des zéros d'intermittence. Les structurels deviennent `NaN`, rien n'est imputé. | Un zéro de fermeture n'est pas une observation de demande nulle ; le modèle d'espace d'états sait traiter une valeur manquante, pas un faux zéro. |
| Fériés transférés | Un férié `transferred=True` n'est pas chômé ; le jour chômé est la ligne `Transfer`. Flags nationaux, régionaux (province du magasin), locaux (ville). | Le creux de ventes suit le jour chômé ; un décalage d'un jour est l'échelle d'une campagne. |
| Séisme | Fenêtre de six semaines à partir du 16 avril 2016, dummies hebdomadaires, flag régional pour Manabí et Esmeraldas ; **exclus des régresseurs** de campagne (ils absorberaient tout effet voisin). | Covariable pour la lecture, pas pour l'estimation des campagnes. |
| Paie bimensuelle, saisonnalité, pétrole | Dummies du 15 et de fin de mois (+ lendemains), jour de la semaine, gradient hebdomadaire du log du pétrole (forward-fill des week-ends). | Le pétrole bouge au mois ; seul son gradient peut compter à l'échelle d'une fenêtre de trois semaines. |
| Transformation | `log1p` (alternative `asinh`) ; effets en unités recalculés trajectoire par trajectoire. | Effets multiplicatifs ; pas de biais de retransformation. |
| Traitement | `onpromotion` est une intensité avec un fond permanent : une **campagne** est un saut au-dessus de la médiane glissante (60 jours), d'une marge qui dépasse le bruit de Poisson, pendant au moins trois jours actifs, précédé de 120 jours calmes (moins de 10 % de jours de campagne, aucun dans les 30 derniers), sans zéro structurel dans la fenêtre, suivi de 14 jours d'horizon. Statut inconnu avant la première promotion du fichier. | Règle fixée avant les résultats ; le nombre de campagnes qui y survivent est un résultat. |
| Pool de contrôle | Registre : autre famille du même magasin sans arête de substitution déclarée, ou même famille d'un autre magasin du cluster ou de la province (pas de la même ville) ; non promue dans la fenêtre ; sans zéro structurel ; volume médian ≥ 5 ; corrélation de pré-période seulement pour plafonner à 12. | L'ajustement n'est jamais un critère d'admission : le meilleur ajustement est souvent le substitut le plus proche. |

## Le monde synthétique

`dgp.py` écrit les cinq fichiers de Kaggle avec les mêmes pièges (25 décembre absent, magasin ouvert tard, famille
jamais vendue, fermetures du 1er janvier, férié transféré, lignes du séisme, promotions permanentes sans effet) et
un contrefactuel **connu** (campagnes plantées avec réponse par famille, creux de *pull-forward*, cannibalisation
d'un substitut déclaré). Il sert aux tests (`uv run pytest`) et a servi à développer toute la chaîne avant
l'arrivée des fichiers Kaggle (`nr ingest --source synthetic`).

## Résultats

Les chiffres de ce bloc sont dans `data/processed/manifest.json` et `data/results/campaigns.json` ; le README
racine les reprend une fois la chaîne exécutée sur les fichiers Kaggle.
