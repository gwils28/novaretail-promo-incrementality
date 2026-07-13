# NovaRetail : l'incrémentalité causale des promotions, mesurée et surveillée

Projet d'application (22 juin → 27 juillet 2026) de la veille
[*L'impact causal en séries temporelles, le contrefactuel et ses placebos*](https://gwils28.github.io/posts/impact-causal-series-temporelles/)
(dépôt [`time_series_causal_statistics`](https://github.com/gwils28/time_series_causal_statistics), package
`causal_ts`, réutilisé ici en dépendance épinglée). Client fictif : **NovaRetail**, chaîne d'épicerie qui multiplie
les promotions sans connaître leur incrémentalité réelle. Données : le panel Favorita de la compétition Kaggle
[*Store Sales*](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) (54 magasins × 33 familles
× 1 684 jours). Le cahier des charges et les décisions de cadrage sont dans [`plan.md`](plan.md).

Le fil du projet : **un contrefactuel est une prévision ; son contenu causal tient à un design** (quelles
campagnes, quels donneurs, quels placebos), et ce design se note sur vérité connue avant de produire un chiffre.

## Ce que je corrige par rapport au cahier des charges

| Le cahier des charges dit | Ce que les données et les vérifications disent | Bloc |
|---|---|---|
| « Favorita n'inclut pas les zéros, décider de les reconstruire » | Le fichier *Store Sales* est dense (une ligne par magasin × famille × jour, sauf le 25 décembre). La décision n'est pas de reconstruire mais de **distinguer les zéros structurels** (magasin pas ouvert, famille jamais vendue, fermeture), mis à `NaN`, des zéros d'intermittence, gardés. | [1](01_data/README.md) |
| « Courbe d'élasticité-prix par famille » | Aucun prix dans Favorita ; `onpromotion` compte des articles. Le livrable devient une **réponse à l'intensité promotionnelle** par Double ML, nommée ainsi, avec ses hypothèses propres. | [3](03_estimation/README.md) |
| « Traitement : `onpromotion` > 0 » | La plupart des familles sont promues en permanence à bas niveau. Le traitement est **construit** : une campagne est un saut au-dessus du fond avec une pré-période calme et un horizon de *pull-forward*. La part des jours promus qui sont mesurables par ce design est petite, et c'est le premier résultat. | [1](01_data/README.md), [2](02_identification/README.md) |
| « Familles non promues comparables = pool de contrôle » | C'est le cas que la veille a montré biaisé (+26 % avec un donneur cannibalisé, placebo dans le temps vert). Le pool est un **registre** : arêtes de substitution déclarées, pas de promotion dans la fenêtre, corrélation seulement pour plafonner ; puis isolation par placebo dans l'espace. | [2](02_identification/README.md) |
| « Intervalle crédible » | Intervalles de prévision fréquentistes (MLE, trajectoires simulées, tirages de paramètres), notés par leur **couverture mesurée** sur vérité plantée dans le panel réel. Les estimateurs à erreurs iid (OLS, SC, DiD) sont notés de la même façon. | [4](04_backtesting/README.md) |
| « Séisme : choc exogène parfait » | Banc d'essai de la méthode, pas une promotion. Les donneurs (provinces éloignées) ont eux-mêmes vendu plus (aide d'urgence) : le placebo dans l'espace le montre, l'effet est une borne basse. | [3](03_estimation/README.md) |
| « Stack : Airflow/Dagster, tfcausalimpact, EconML/DoubleML, statsforecast, MLflow, versioning » | Dagster en local, MLflow en registre d'**estimations** (chaque campagne réajuste son modèle), manifeste sha256 des données, DoubleML en contre-vérification du DML maison. Pas de `tfcausalimpact` (a priori par défaut trop étroit, mesuré dans la veille), pas de `statsforecast`, pas de cloud. | [5](05_pipeline/README.md) |
| « Uplift prédit vs marge réalisée » | Pas de marge dans Favorita : la réconciliation est démontrée sur un grand livre simulé avec une dérive injectée, et le code est celui qu'un vrai grand livre alimenterait. | [6](06_monitoring/README.md) |

## Résultats

*Section remplie à partir de `data/results/*.json` une fois la chaîne exécutée sur les fichiers Kaggle.*

## Blocs

| Bloc | Dossier | Commandes | Contenu |
|---|---|---|---|
| 1 Données | [`01_data`](01_data/README.md) | `nr ingest`, `nr campaigns` | panel dense, zéros structurels, calendrier (fériés transférés, paie, séisme, pétrole), détection de campagnes, registre de donneurs, monde synthétique |
| 2 Identification | [`02_identification`](02_identification/README.md) | | les trois hypothèses, qui les voit, ce que le registre prévient et ce que l'isolation détecte |
| 3 Estimation | [`03_estimation`](03_estimation/README.md) | `nr estimate`, `nr dml`, `nr quake` | contrefactuel BSTS par campagne, effets en unités pendant/après/net, SC et SC augmenté, DiD, DML, séisme |
| 4 Backtesting | [`04_backtesting`](04_backtesting/README.md) | `nr backtest` | vérité plantée sur le panel réel, couverture, faux positifs, puissance, formes, pull-forward, cannibalisation |
| 5 Pipeline | [`05_pipeline`](05_pipeline/README.md) | `dagster dev` | assets Dagster, MLflow, manifeste de données |
| 6 Monitoring | [`06_monitoring`](06_monitoring/README.md) | `nr monitor`, `nr calendar` | quatre signaux, réconciliation de marge, calendrier à budget constant |

## Notebooks

Matériel pédagogique, en français : chaque notebook construit l'intuition, lance l'expérience depuis le package
et lit le résultat, avec des encarts fixes (**Intuition**, **Exemple**, **Attention**, **À retenir**). Ils sont
commités exécutés et exportent les figures de [`figures/`](figures). Sources en format pourcent dans
[`notebooks/src/`](notebooks/src).

| Notebook | Ce qu'il explique |
|---|---|
| [`01_donnees_panel_zeros`](notebooks/01_donnees_panel_zeros.ipynb) | densité du panel, zéros structurels vs intermittents, fériés transférés, paie, pétrole, `onpromotion` comme intensité |
| [`02_identification_campagnes_donneurs`](notebooks/02_identification_campagnes_donneurs.ipynb) | hypothèses et angles morts, graphe d'identification, détection d'une campagne, raisons de refus, registre d'une campagne |
| [`03_estimation_une_campagne`](notebooks/03_estimation_une_campagne.ipynb) | une campagne pas à pas : ajustement de pré-période, contrefactuel, unités, placebos, isolation, sensibilité au pool, six estimateurs, verdict |
| [`04_backtesting_verite_plantee`](notebooks/04_backtesting_verite_plantee.ipynb) | couverture, largeur, faux positifs à effet nul, puissance, formes, pull-forward, cannibalisation ; relecture des vraies campagnes |
| [`05_reponse_intensite_dml`](notebooks/05_reponse_intensite_dml.ipynb) | pourquoi pas d'élasticité-prix, partialling-out, θ avec et sans retards, holdout, DoubleML, par famille, dose-réponse, concordance |
| [`06_seisme_banc_essai`](notebooks/06_seisme_banc_essai.ipynb) | géographie et donneurs déclarés, effet semaine par semaine, placebos et interférence, par famille |
| [`07_monitoring_calendrier`](notebooks/07_monitoring_calendrier.ipynb) | les quatre signaux dans le temps, réconciliation propre vs dérive injectée, réallocation du calendrier |

## Reproduire

```bash
make install                      # uv sync (Python 3.13), causal_ts en dépendance git épinglée
make test-fast                    # tests unitaires sur le monde synthétique
# déposer store-sales-time-series-forecasting.zip (Kaggle) dans data/raw/, puis :
uv run nr ingest && uv run nr campaigns
uv run nr estimate --param-draws 40          # registre, contrefactuel, placebos, isolation, verdict par campagne
uv run nr backtest --n 30                    # vérité plantée, scores par estimateur
uv run nr quake && uv run nr dml             # séisme, réponse à l'intensité
uv run nr monitor && uv run nr calendar      # signaux, réconciliation simulée, réallocation
make notebooks                               # ré-exécute les sept notebooks
```

Sans fichiers Kaggle : `uv run nr ingest --source synthetic` exécute la même chaîne sur le monde synthétique.
