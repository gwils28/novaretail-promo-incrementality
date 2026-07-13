# Bloc 2 · Identification : le design avant l'algorithme

Notebook : [02](../notebooks/02_identification_campagnes_donneurs.ipynb). Code : `donors.py`, `campaigns.py`.

## Les trois hypothèses et qui les voit

| Hypothèse | Violation plausible chez NovaRetail | Test qui peut la voir | Test aveugle |
|---|---|---|---|
| Date de campagne « comme aléatoire » sachant le passé | campagne lancée parce que les ventes baissaient | placebo dans le temps (si la tendance propre existe déjà en pré-période) | placebo dans l'espace |
| Contrôles non affectés (SUTVA) | cannibalisation d'un substitut, halo d'un complément, client qui change de magasin | placebo dans l'espace, isolation *leave-one-out* | **placebo dans le temps** (vert par construction) |
| La relation contrôles ↔ cible persiste | changement d'assortiment, rupture le jour de la campagne | aucun test avant coup ; réconciliation de marge après | tous |

La veille (`time_series_causal_statistics`, notebook 03) a mesuré ces angles morts sur panels à vérité connue :
contrôle cannibalisé à 60 % du lift → biais +26 %, placebo dans le temps vert dans 90 % des cas ; rupture de
relation → biais −11 %, couverture 0,56, rien ne se déclenche.

## Ce que le registre prévient et ce que l'isolation détecte

- **Prévention** : arêtes de substitution déclarées entre familles (`SUBSTITUTION_EDGES`, 24 paires), même ville
  refusée pour les donneurs de même famille, promotion dans la fenêtre refusée.
- **Détection** : placebo dans l'espace à la vraie date, chaque donneur traité à son tour avec les autres et le
  calendrier comme régresseurs ; retrait du donneur à la plus petite probabilité de queue, puis rejeu, jusqu'à ce
  que rien ne soit signalé. Le nombre de tours est un signal de monitoring.
- **Ce qui reste invisible** : une cannibalisation entre deux familles non liées par une arête et trop faible
  pour que le placebo dans l'espace la voie (sa puissance est modeste), et une relation qui casse à la date.

## Le traitement n'est pas binaire

`onpromotion` compte des articles. Une campagne est un saut au-dessus du fond (règle du bloc 1) ; son intensité
est le compte ramené au quantile 95 % des comptes positifs de la famille (bloc 3, DML). L'effet d'une campagne est
donc l'effet d'un **saut d'intensité à l'intensité observée**, pas l'effet d'« être en promotion ».
