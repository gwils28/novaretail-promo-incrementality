# Bloc 3 · Estimation : contrefactuel par campagne, contrôle synthétique, double ML

Code : `estimate.py`, `backtest.py` (estimateurs), `dml.py`, `earthquake.py`. Commandes : `nr estimate`,
`nr dml`, `nr quake`. Notebooks : [03](../notebooks/03_estimation_une_campagne.ipynb),
[05](../notebooks/05_reponse_intensite_dml.ipynb), [06](../notebooks/06_seisme_banc_essai.ipynb).

## Le modèle de référence

`causal_ts.bsts.causal_impact` (veille) : niveau local + régression statique sur les donneurs (échelle `log1p`)
et les régresseurs de calendrier, maximum de vraisemblance sur la pré-période, contrefactuel par filtre de Kalman,
intervalles par trajectoires simulées conjointement, 40 tirages de paramètres pour l'estimation finale. Les effets
en unités sont lus trajectoire par trajectoire et scindés : **pendant** la campagne (le lift), **après** (le creux
de *pull-forward* sur 14 jours), **net**.

## Les autres estimateurs, sur la même interface

| Estimateur | Ce qu'il suppose | Son intervalle |
|---|---|---|
| `bsts_llevel` | niveau qui dérive + relation statique | trajectoires simulées (état + observation, paramètres en option) |
| `bsts_static` | relation statique, erreurs iid | analytique |
| `ols` | idem, prédiction OLS | t de Student, iid |
| `synthetic_control` | poids convexes sur les donneurs | iid sur les résidus de pré-période (la méthode classique n'en donne pas) |
| `augmented_sc` | poids convexes + correction ridge du déséquilibre de pré-période (Ben-Michael, Feller, Rothstein 2021) | iid, idem |
| `diff_in_diff` | la cible suit la moyenne des donneurs à un écart constant près | iid |

Aucun n'est choisi sur sa largeur : le bloc 4 les note par leur couverture sur vérité plantée.

## Le verdict de crédibilité (protocole, `plan.md` § 4)

R² un pas en avant ≥ 0,5 ; couverture de l'intervalle un pas dans [0,90 ; 0,98] ; Ljung-Box p ≥ 0,01 ; au plus
2 fausses dates sur 8 dont l'intervalle exclut zéro ; isolation des donneurs signalés. Une campagne qui échoue est
« non mesurable » : le design ne sait pas lire cette série, ce qui ne dit rien de la campagne.

## Réponse à l'intensité (DML), pas élasticité-prix

Favorita n'a pas de prix. Modèle partiellement linéaire `log1p(ventes) = θ·intensité + g(contrôles) + ε`,
nuisances par gradient boosting, validation croisée par **blocs de mois**, erreur standard HC. Spécifications
avec et sans retards de la série, holdout sur la dernière année, contre-vérification DoubleML, θ par famille et
dose-réponse par paliers. L'hypothèse d'identification (intensité comme aléatoire sachant les contrôles) est
plus forte que celle du contrefactuel par campagne ; la concordance entre les deux designs est le seul lien.

## Le séisme

Banc d'essai de la méthode, pas une promotion : provinces frappées contre donneurs déclarés (provinces éloignées,
Quito et Guayaquil exclues comme pôles de collecte). Le placebo dans l'espace sur les magasins donneurs montre
l'interférence des achats d'aide d'urgence ; l'effet sur les provinces frappées est alors une borne basse.

## Résultats (fichiers Kaggle)

### Séisme (`nr quake`, `data/results/quake.json`)

| Mesure | Valeur |
|---|---|
| Unités traitées / donneurs déclarés | magasins 43 (Esmeraldas), 53 et 54 (Manabí) / 20 magasins des provinces éloignées |
| Effet sur 8 semaines, en unités | **+39 %** [+32 % ; +46 %], soit +572 000 unités |
| Semaine par semaine | +10 % la première semaine, +33 % la deuxième, puis **+45 à +55 % jusqu'à la huitième** : l'effet ne s'estompe pas dans la fenêtre |
| Placebo dans le temps | 1 fausse date sur 4 se déclenche |
| Placebo dans l'espace | **5 magasins donneurs sur 20** ont un effet propre qui exclut zéro : l'interférence des achats d'aide d'urgence est visible, l'effet sur les provinces frappées est une borne basse |
| Par famille (4 semaines) | toutes les familles de volume montent, GROCERY I le moins (+3 %), LINGERIE, SEAFOOD, BEAUTY le plus (+28 à +78 %) |

**Lecture.** La méthode trouve un effet massif et daté. Mais sa forme (croissant, pas transitoire) ne ressemble pas
à une ruée d'une semaine : magasins concurrents endommagés, population déplacée vers les villes des magasins,
ou relation donneurs-cible qui a changé le 16 avril. Aucun test ne tranche ; sur une campagne, seul le rapprochement
avec la marge réalisée le ferait. C'est exactement l'angle mort que la veille avait nommé.

### Réponse à l'intensité (`nr dml`, `data/results/dml.json`, 400 000 lignes tirées au sort)

| Spécification | θ (log-ventes par unité d'intensité) |
|---|---|
| pooled, retards + mois + tendance | 0,32 ± 0,006 |
| pooled, sans retards | 0,59 ± 0,007 |
| entraînement (< août 2016) / holdout (≥ août 2016) | 0,34 / 0,36 |
| DoubleML (bibliothèque, 200 000 lignes) | 0,38 ± 0,009 |
| sans mois ni tendance (première passe) | 0,15 à 0,25 |

θ bouge de 0,15 à 0,59 selon ce que les nuisances contiennent. Les erreurs standard sont minuscules, la
**sensibilité à la spécification** ne l'est pas : la date d'une promotion n'est pas « comme aléatoire » sachant des
contrôles de calendrier, et le DML ne peut pas le réparer. La dose-réponse par paliers est monotone (0,15 → 0,31 du
palier (0,2 ; 0,5] au palier > 1). Par famille, les θ de PRODUCE (5), SCHOOL AND OFFICE SUPPLIES (2,5) et LAWN AND
GARDEN (1,4) ne sont pas des réponses : régime d'enregistrement de `onpromotion` (PRODUCE : ~200 un jour par
semaine) ou promotions calées sur le pic saisonnier. Le calendrier n'utilise pas ces θ ; il utilise les campagnes.
