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

Chiffres de `data/results/*.json`, produits par le CLI sur les fichiers Kaggle (détail et lecture dans chaque bloc).

| Question | Réponse mesurée |
|---|---|
| Combien de promotions sont mesurables ? | **1 592 campagnes** sur 67 458 sauts de `onpromotion` (2,4 %) : la plupart des sauts sont trop courts ou tombent dans une pré-période déjà promue. 301 ont été estimées (stratifiées par famille), **170 passent le protocole** de crédibilité (56 %). |
| Quel est l'uplift d'une campagne ? | Lift médian pendant la campagne **+1,2 %** (déciles 10 et 90 : −8 % et +13 %). 7 % des campagnes ont une borne basse positive pendant la campagne, 5 % sur le net (campagne + 14 jours) ; 1 à 2 % ont une borne haute négative. Somme nette sur les 170 campagnes : **−0,6 %** du contrefactuel. |
| Y a-t-il du *pull-forward* ? | Non mesurable à ce niveau : la part médiane des unités reprises après la campagne est −14 % (un léger surplus après, pas un creux), avec des intervalles qui contiennent zéro presque partout. |
| Le design est-il fiable ? | Sur vérité plantée dans les vraies séries : couverture à 95 % de **0,83 à 0,90** pour le niveau local, 0,5 à 0,7 pour OLS, contrôle synthétique, DiD ; **17 % de faux positifs** à effet nul ; puissance 20 % pour +5 %, 40 % pour +15 %. Le protocole rejette un quart des campagnes par le placebo dans le temps, à ce taux, par construction. |
| Le séisme ? | **+39 %** [+32 ; +46] en unités sur 8 semaines dans les provinces frappées, effet croissant et non transitoire ; 5 donneurs sur 20 bougent avec le choc (achats d'aide d'urgence) : l'effet est une borne basse et l'interférence est visible parce qu'on sait où regarder. |
| La réponse à l'intensité ? | θ pooled de **0,15 à 0,59** selon la spécification des nuisances (retards, mois, tendance) ; holdout 0,36, DoubleML 0,38 ; dose-réponse monotone ; **concordance nulle** (corrélation 0,02) avec les lifts des 170 campagnes mesurées. L'instabilité est le résultat : la date d'une promotion n'est pas « comme aléatoire » sachant un calendrier. PRODUCE (θ = 5) est un artefact d'enregistrement. |
| Le monitoring ? | Le placebo roulant sur du vide est en alerte sur deux tiers de la période (> 10 %) ; l'alarme de réconciliation se déclenche trois mois après une dérive de 2 %/mois injectée, avec une fausse alarme sur 44 mois. |
| Le calendrier ? | 48 jours déplacés sur 2 327 (2 %), **64 % du budget non dépensé** sur la borne basse : presque aucune campagne ne vaut ses jours à ce niveau de preuve. |

### Recul sincère

- **Le soupçon de la direction tient, mais la preuve est faible.** La somme nette des 170 campagnes crédibles est
  nulle à −0,6 %, et seules 7 % ont un lift dont la borne basse dépasse zéro. Mais le même design ne détecte
  qu'une fois sur cinq un lift de 5 % planté dans ces séries. « Pas d'effet mesurable » et « pas d'effet » sont
  deux phrases différentes ; ce projet n'autorise que la première.
- **Les vraies séries coûtent dix points de couverture.** La veille mesurait 0,92 à 0,94 sur panels synthétiques
  pour le niveau local ; ici 0,83 à 0,90, et 17 % de faux positifs. Zéros, valeurs extrêmes, changements de régime
  d'enregistrement : la différence entre un panel propre et un fichier.
- **Le traitement était le problème, pas l'estimateur.** `onpromotion` est un compte avec un fond permanent, un
  pic hebdomadaire sur PRODUCE, des régimes qui changent en 2016. La moitié du temps du projet est allée à décider
  ce qu'est un zéro et ce qu'est une campagne ; c'est là que se jouait la validité, pas dans le choix entre BSTS et
  contrôle synthétique (qui, lui, se tranche par la couverture : niveau local).
- **Ce que la méthode ne voit pas**, et que rien ici ne peut voir : une relation donneurs-cible qui casse à la
  date de campagne. Le séisme en donne peut-être un exemple (effet croissant sur huit semaines). Seul un
  rapprochement avec une marge réalisée, ou un **holdout réel** sur quelques campagnes, tranche. C'est la
  recommandation principale à NovaRetail : mesurer moins de campagnes, mais en ancrer quelques-unes par
  l'expérimentation.
- **Les θ du DML ne sont pas des élasticités**, et leur sensibilité à la spécification dit que l'hypothèse
  « intensité comme aléatoire sachant les contrôles » n'est pas tenable sur ce panel. Ils sont rapportés pour
  cela, pas pour le calendrier.
- **Taille des études** : 30 fenêtres par cellule du backtesting (±0,1 sur chaque couverture), 300 campagnes sur
  1 592 (13 minutes de calcul), un seul tirage de la stratification. Les ordres de grandeur sont stables d'une
  exécution à l'autre ; la deuxième décimale ne l'est pas.

## En un paragraphe

Le traitement a dû être construit (un saut de `onpromotion` au-dessus de son fond, avec une pré-période calme), le
pool de contrôle a été un registre de règles plutôt qu'un choix par l'ajustement, et les zéros ont été triés en
cinq sortes avant la moindre estimation. Sur ces bases, le contrefactuel à niveau local de la veille, noté par sa
couverture sur vérité plantée dans les vraies séries (0,83 à 0,90, contre 0,5 à 0,7 pour les estimateurs à
erreurs iid), mesure 170 campagnes crédibles dont la somme nette est nulle, avec une puissance qui ne permet pas de
dire que les petites promotions ne font rien. Le séisme montre la méthode à l'œuvre sur un choc réel et montre
l'interférence des donneurs ; le DML montre ce qui arrive quand l'hypothèse d'identification ne tient pas ; le
monitoring montre que le seul signal qui voit une relation qui casse est la marge réalisée. La recommandation à
NovaRetail est de mesurer moins et d'expérimenter un peu : quelques campagnes avec magasins témoins tirés au sort
ancreraient tout le reste.

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
