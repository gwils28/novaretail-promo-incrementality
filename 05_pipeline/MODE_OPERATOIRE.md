# Mode opératoire · Dagster et MLflow en local

Ce guide lance le pipeline (`pipeline/definitions.py`), suit ses runs dans Dagster et consulte les estimations
et le monitoring enregistrés dans MLflow, par l'interface ou par l'API. Tout tourne en local, sans Docker.

## 1. Installation (une fois)

```bash
cd TimesSeriesCausal_NovaRetail
uv sync --group pipeline          # dagster, dagster-webserver, mlflow
```

Les fichiers Kaggle doivent être dans `data/raw/` (voir le README racine).

## 2. Variables d'environnement (à chaque terminal)

```bash
export DAGSTER_HOME=$PWD/.dagster && mkdir -p $DAGSTER_HOME   # historique des runs conservé entre sessions
export MLFLOW_DISABLE_AGENT_HINT=1                           # coupe un message d'aide de MLflow 3
```

Sans `DAGSTER_HOME`, Dagster écrit dans un dossier temporaire et l'historique est perdu à l'arrêt.

MLflow utilise un backend SQLite (`mlflow.db` à la racine) et range les artefacts sous `mlruns/` ; les deux sont
ignorés par git. MLflow 3 refuse l'ancien backend fichier. Pour pointer ailleurs (serveur MLflow distant par
exemple) : `export MLFLOW_TRACKING_URI=http://hote:5000`.

## 3. Dagster : lancer et suivre le pipeline

Terminal 1 :

```bash
uv run dagster dev -f pipeline/definitions.py     # http://localhost:3000
```

| Onglet | Ce qu'on y voit |
|---|---|
| Assets, vue graphe | la chaîne `panel -> campaigns -> estimates / backtest -> monitoring / promo_calendar` |
| Un asset, Materializations | les métadonnées de chaque run : campagnes mesurables, taux de faux positifs, `mlflow_run_id` |
| Checks | le garde-fou `placebo_guard` : échoue si le taux de faux positifs sur fenêtres jamais traitées dépasse 10 % |
| Runs | journaux et durée de chaque exécution |
| Automation | la planification `weekly_refresh` (lundi 6 h), à activer à la main ; elle ne tourne que si `dagster dev` tourne |

Matérialiser depuis la ligne de commande :

```bash
uv run dagster asset materialize -f pipeline/definitions.py --select monitoring   # quelques secondes
uv run dagster asset materialize -f pipeline/definitions.py --select '*'          # chaîne complète
```

Une sélection partielle (`--select monitoring`) relit les fichiers déjà présents dans `data/results/` sans relancer
l'amont.

### Paramètres de l'estimation

La configuration par défaut de `estimates` fait 0 tirage de paramètres ; les chiffres des README viennent de
`--param-draws 40` (environ 13 minutes). Pour les reproduire, ouvrir le Launchpad (Maj + clic sur Materialize) et
coller :

```yaml
ops:
  estimates:
    config: {param_draws: 40, max_campaigns: 300}
```

**Attention** : matérialiser `estimates`, `backtest` ou `monitoring` réécrit les `data/results/*.json` versionnés.
Après un essai avec d'autres paramètres, `git restore data/results/*.json` rétablit les résumés publiés.

## 4. MLflow : consulter les estimations et le monitoring

Terminal 2 :

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000   # http://localhost:5000
```

Expérience `novaretail-promo-incrementality`, un run par matérialisation :

| Run | Paramètres | Métriques | Artefacts |
|---|---|---|---|
| `estimates` | pool, tirages de paramètres, trajectoires, graine | contenu numérique de `estimates.json` | `estimates.json`, `estimates.parquet`, `donors.parquet` |
| `backtest` | campagnes plantées par spécification, graine | contenu numérique de `backtest.json` | `backtest.json`, `planted_scores.parquet` |
| `monitoring` | graine | `fpr_never_treated`, `blocked_by_fit`, alarmes de réconciliation | `monitor.json`, `rolling_fpr.parquet`, `reconciliation_drift.parquet` |

Pour comparer deux runs : les cocher puis **Compare**. Le registre contient des estimations et leurs contrôles de
validité, pas des modèles : chaque campagne réajuste son propre contrefactuel.

## 5. Consulter par l'API

Les deux serveurs doivent tourner (étapes 3 et 4).

MLflow, API REST :

```bash
curl -s -X POST localhost:5000/api/2.0/mlflow/runs/search \
  -H 'Content-Type: application/json' -d '{"experiment_ids":["1"]}'
```

MLflow, depuis Python (sans serveur) :

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.search_runs(experiment_names=["novaretail-promo-incrementality"])
```

Dagster, API GraphQL (un explorateur interactif est servi sur http://localhost:3000/graphql) :

```bash
curl -s localhost:3000/graphql -H 'Content-Type: application/json' \
  -d '{"query":"{ assetNodes { assetKey { path } } }"}'
```

## 6. Dépannage

| Symptôme | Cause et remède |
|---|---|
| `The filesystem tracking backend ... is in maintenance mode` | un ancien `file://mlruns` est configuré : retirer `MLFLOW_TRACKING_URI` ou le passer en `sqlite:///mlflow.db` |
| Historique Dagster vide à chaque lancement | `DAGSTER_HOME` non exporté dans ce terminal |
| `mlflow_run_id: mlflow not installed` dans les métadonnées | le groupe `pipeline` n'est pas installé : `uv sync --group pipeline` |
| Port déjà utilisé | `dagster dev -p 3001`, `mlflow ui --port 5001` |
| `placebo_guard` en échec | comportement attendu sur Favorita (17 % de faux positifs, voir le bloc 6) |
