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

## Résultats (fichiers Kaggle, `nr ingest`, `nr campaigns`)

| Mesure | Valeur |
|---|---|
| Lignes du fichier / attendues (54 × 33 × 1 688) | 3 000 888 / 3 008 016 ; les 7 128 lignes manquantes sont les quatre 25 décembre |
| Part de lignes à ventes nulles dans le fichier | 31 % |
| … dont structurelles après les cinq flags | **80 %** (magasin pas ouvert, famille jamais vendue : 53 couples, fermetures : 822 jours-magasin, famille pas encore référencée : 595 séries soit 9 % des lignes, séries de zéros ≥ 28 jours : 6 % des lignes) |
| Zéros d'intermittence gardés comme observations | 6 % des lignes du panel |
| Magasins ouverts en cours de période | 8 (dont le 52 en avril 2017) |
| Sauts de `onpromotion` détectés / campagnes mesurables | 67 458 / **1 592** (2,4 %) ; refus : 63 302 trop courts, 802 pré-période occupée, 639 zéros structurels, 610 trop longs, 513 promotion dans les 30 jours précédents |
| Durée des campagnes mesurables (déciles 10, 50, 90) | 3, 6, 20 jours ; 1 087 campagnes sur 1 592 ont un horizon tronqué par la campagne suivante |

**Ce que les fichiers ont imposé.** Le cahier des charges prévoyait une décision sur des zéros absents ; les
fichiers ont demandé l'inverse, trier des zéros présents : famille pas encore référencée dans le magasin (les
séries commencent à des dates différentes selon la famille), trimestres entiers à zéro au milieu d'une série
(PRODUCE, 2014 et 2015), et un `onpromotion` qui, pour PRODUCE, vaut ~200 un jour par semaine et 0 ou 1 les
autres jours. Chacun de ces cas, pris pour une observation, fabrique un contrefactuel faux ; chacun a sa règle,
déclarée dans `data.py`, et sa part dans le tableau ci-dessus.
