# Bloc 5 · Pipeline : Dagster en local, MLflow en registre d'estimations

Code : `pipeline/definitions.py`, `src/novaretail/cli.py`, `workspace.py`.

```bash
uv sync --group pipeline                                   # dagster, dagster-webserver, mlflow
uv run dagster dev -f pipeline/definitions.py              # UI sur http://localhost:3000
uv run dagster asset materialize -f pipeline/definitions.py --select '*'
uv run mlflow ui --backend-store-uri file://$PWD/mlruns
```

## Les assets

```
panel ──> campaigns ──> estimates ──> monitoring
                   └──> backtest ──┘        └──> (asset check : placebo_guard)
                        estimates ──> promo_calendar
```

Chaque asset appelle une commande du CLI et écrit les mêmes fichiers sous `data/results/` : un notebook, le CLI
et le pipeline lisent une seule source de vérité. La planification hebdomadaire (`weekly_refresh`, lundi 6 h)
rejoue la chaîne.

## Ce que MLflow enregistre

Pas des modèles : chaque campagne réajuste son propre contrefactuel. Le registre contient, par run, les
paramètres (plafond du pool, tirages de paramètres, nombre de trajectoires, graine), les métriques résumées
(`estimates.json`) et la table des estimations avec leurs contrôles de validité en artefact. C'est un registre
d'**estimations et de leurs vérifications**.

## Versioning des données

`data/processed/manifest.json` : sha256, taille et nombre de lignes de chaque fichier brut, plus le rapport de
construction du panel. Un run est reproductible à manifeste égal ; DVC n'est pas nécessaire à cette échelle.

## Temps de calcul (fichiers Kaggle, portable 8 cœurs)

| Étape | Durée |
|---|---|
| `nr ingest` + `nr campaigns` | 2 s + 30 s |
| `nr estimate --max 300 --param-draws 40` | 13 min (2,2 s par campagne : registre, isolation, contrefactuel, 8 placebos) |
| `nr backtest --n 30` | 2 min |
| `nr quake`, `nr dml --sample 400000` | 15 s, 3 min |
| `make notebooks` | 10 min |

Les 1 592 campagnes mesurables tiendraient en une heure ; la planification hebdomadaire n'estime que les
campagnes nouvelles.

## Ce qui n'est pas fait

Pas de Docker, pas de cloud, pas d'Airflow, pas de `statsforecast` (les baselines de prévision pure n'apportent
rien à la question causale), pas de `tfcausalimpact` (la veille a montré que son a priori par défaut donne des
intervalles deux fois trop étroits ; `causal_ts` le remplace).
