# NovaRetail : incrémentalité causale des promotions — cahier des charges et cadrage

Projet d'application de la veille [*L'impact causal en séries temporelles, le contrefactuel et ses placebos*](https://gwils28.github.io/posts/impact-causal-series-temporelles/) (dépôt `gwils28/time_series_causal_statistics`, package `causal_ts`, réutilisé ici comme dépendance épinglée). Méthode de travail : [*Claude Code, méthode révisée*](https://gwils28.github.io/posts/claude-code-methode-revisee/). Démarrage le 22 juin 2026, 36 jours.

## 1. Le cahier des charges (livrable 2, tel que reçu)

**Client fictif « NovaRetail »**, chaîne d'épicerie omnicanale (≈ 50 magasins + e-commerce) qui multiplie les promotions (remises, BOGO) sans connaître leur incrémentalité causale réelle. Soupçon de la direction : une large part des ventes « en promo » auraient eu lieu de toute façon (aubaine), avec en prime du *pull-forward* et de la cannibalisation entre familles. Si 30 à 40 % du budget promo est non incrémental, le réallouer change la marge. Il faut construire le **contrefactuel** (ventes sans promo), isoler l'uplift causal par campagne, puis nourrir l'optimisation du calendrier et du pricing.

**Données** : Kaggle, *Store Sales — Time Series Forecasting* (Corporación Favorita, Équateur). Journalier, magasin × famille, `onpromotion` = nombre d'articles de la famille en promotion ce jour-là dans ce magasin ; métadonnées magasin, prix du pétrole, fériés (avec `transferred`) ; séisme du 16 avril 2016 comme choc exogène.

**Périmètre demandé**
- A. Préparation : panel magasin × famille × jour, décision explicite sur les zéros, calendrier (fériés transférés, séisme), covariables (pétrole et lags, saisonnalités, paie bimensuelle), transformation `log1p`/`asinh`, définition du traitement et du pool de contrôle.
- B. Modélisation : identification d'abord (SUTVA face à la cannibalisation, contrôles non affectés, tendances parallèles) ; baseline CausalImpact (BSTS) ; avancé : contrôle synthétique / augmenté, Double ML pour l'élasticité, DiD en robustesse. Livrable : uplift par campagne avec intervalle, courbe d'élasticité par famille.
- C. Backtesting : placebo dans le temps et dans l'espace, qualité du fit pré-période, sensibilité au pool, couverture réelle des intervalles, tendances parallèles, holdout pour l'élasticité.
- D. Stack : orchestration (Airflow/Dagster), `tfcausalimpact`/`pycausalimpact`, EconML/DoubleML, `statsforecast`, MLflow, versioning des données.
- E. Monitoring : dérive de la relation contrôle↔cible, intégrité des contrôles, placebos périodiques, réconciliation uplift prédit vs marge réalisée.

## 2. Décisions de cadrage (interview du 22 juin)

| Question | Décision |
|---|---|
| Accès aux données | Pas d'identifiants Kaggle sur la machine ; l'auteur dépose `store-sales-time-series-forecasting.zip` dans `data/raw/`. Le script d'ingestion l'attend ; en attendant, un panel synthétique « façon Favorita » à vérité connue sert aux tests et aux études de couverture. |
| Langue | READMEs, notebooks, figures en français ; identifiants, docstrings, messages de commit en anglais (cohérence avec `causal_ts`). |
| Git | Commits et pushes faits par l'agent sur ce dépôt (dérogation explicite à la méthode), antidatés du 22 juin au 27 juillet 2026, auteur unique, pas de co-auteur. Dépôt public `gwils28/novaretail-promo-incrementality`. |
| Partie D | Pipeline Dagster exécutable en local (assets : ingestion → features → estimation par campagne → calendrier) et MLflow en backend fichier. Pas de Docker, pas de cloud. Versioning des données par manifestes (hash + lignes + dates) plutôt que DVC. |
| Réutilisation | `causal_ts` (BSTS maison, placebos, diagnostics, baselines, style de figures) en dépendance git épinglée au commit `388c1a3`. On l'étend, on ne le recopie pas. |

## 3. Ce que je corrige ou précise par rapport au cahier des charges

Chaque ligne sera confirmée ou infirmée sur les données ; le README final tiendra le tableau à jour.

1. **Les zéros.** Le cahier des charges décrit la version 2017 du jeu (*Favorita Grocery Sales Forecasting*), qui omet les lignes à ventes nulles. La version 2021 (*Store Sales*) est un panel dense : 54 magasins × 33 familles × 1 684 jours = 3 000 888 lignes, zéros inclus. La vraie décision n'est donc pas « reconstruire les zéros » mais **distinguer les zéros structurels** (magasin pas encore ouvert, famille jamais vendue dans ce magasin, journées de fermeture) des zéros d'intermittence. À vérifier au chargement.
2. **Pas de prix dans Favorita.** Le jeu n'a aucune colonne de prix ; `onpromotion` est un **nombre d'articles en promotion**, pas une remise. Une élasticité-prix au sens strict est inestimable sur ces données. Ce qui l'est : une **courbe de réponse à l'intensité promotionnelle** (ventes en fonction de la part d'articles promus, sous contrôles) par Double ML. Le livrable « élasticité-prix par famille » devient « réponse à l'intensité promo par famille », et le README le dira sans détour.
3. **Le traitement n'est pas binaire.** Une « campagne » n'existe pas dans les données ; il faut la **détecter** : fenêtre où `onpromotion` passe d'un régime bas à un régime haut sur une famille-magasin, précédée d'une pré-période suffisamment propre. Beaucoup de familles sont en promotion quasi permanente à partir de 2015 ; le nombre de campagnes « mesurables » sera bien plus petit que le nombre de jours promus, et c'est une information en soi.
4. **Le séisme n'est pas une promotion.** Il sert de **banc d'essai de la méthode** sur un choc exogène daté, avec les magasins des provinces épargnées comme donneurs et un placebo dans l'espace par province. Son effet n'alimente pas le calendrier promo.
5. **Cannibalisation et pool de contrôle.** Prendre « les familles non promues du même magasin » comme contrôles est exactement le cas que la veille a montré biaisé (+26 % avec un contrôle cannibalisé, placebo dans le temps aveugle). Le pool suit le **registre de donneurs** du bloc 3 de la veille : même magasin ou même cluster, famille différente et sans arête de substitution plausible (liste d'exclusion par paire de familles déclarée avant les résultats), non promue dans la fenêtre, puis isolation leave-one-out par placebo dans l'espace.
6. **« Intervalle crédible ».** Les intervalles du modèle maison sont des intervalles de prévision fréquentistes (MLE, trajectoires simulées, tirages de paramètres en option). Ils sont notés par leur couverture mesurée sur vérité connue, comme dans la veille. `tfcausalimpact` reste une contre-vérification optionnelle (son a priori par défaut donne des intervalles deux fois trop étroits).
7. **Horizon de mesure.** Le cahier des charges parle d'un horizon contrefactuel de 7 à 16 jours ; c'est la fenêtre de la campagne plus un **post-horizon de 14 jours** pour voir le *pull-forward* (creux après la promo). L'uplift net inclut ce creux.
8. **Durée.** 36 jours, pas 1 à 2 mois : `statsforecast`, EconML, Airflow et le versioning DVC ne sont pas implémentés ; la partie D est un pipeline Dagster local, pas une mise en production.

## 4. Protocole fixé avant les résultats

- **Unité d'analyse** : la série journalière `log1p(ventes)` d'une famille dans un magasin. Campagne = fenêtre détectée (§3.3) avec ≥ 120 jours de pré-période où la famille-magasin n'est pas promue (ou promue à un niveau stable et bas), post-période = fenêtre + 14 jours.
- **Modèle de référence** : `causal_ts.bsts.causal_impact` niveau local + régression statique sur les donneurs, saisonnalité hebdomadaire, fériés et paie en régresseurs ; `param_draws=40` pour l'estimation finale.
- **Critère de crédibilité d'une estimation** (sinon la campagne est « non mesurable ») : R² un pas en avant ≥ 0,5 sur la pré-période, couverture un pas en avant dans [0,90 ; 0,98], Ljung-Box p ≥ 0,01, balayage placebo dans le temps sur 8 fausses dates avec au plus 2 intervalles excluant zéro, aucun donneur signalé après isolation.
- **Vérité connue** : promotions plantées sur le panel réel (effet relatif 0 %, +5 %, +15 %, +30 %, formes marche/rampe/impulsion, avec ou sans *pull-forward* et cannibalisation d'un donneur). On mesure couverture, biais et puissance par estimateur (BSTS niveau local, SC convexe, SC augmenté, DiD, OLS). L'estimateur retenu pour le livrable est celui dont la couverture à 95 % est la plus proche du nominal, pas le plus étroit.
- **Taux de faux positifs attendu** : lu sur des séries jamais traitées (placebo dans le temps roulant), pas supposé égal à 5 %.
- **Réponse à l'intensité promo** : DML partiellement linéaire puis par paliers d'intensité, validé par holdout temporel (dernière année) et par la concordance avec les uplifts de campagnes mesurées.
- **Calendrier** : réallocation sous contrainte de budget constant (jours-promo), uplift net borne basse de l'intervalle, interdiction de chevauchement de familles substituables.

## 5. Blocs et livrables

| Bloc | Dossier | Contenu |
|---|---|---|
| 1 Données | `01_data/` | ingestion, panel, zéros structurels, calendrier, covariables, détection de campagnes, registre de donneurs ; notebook 01 |
| 2 Identification | `02_identification/` | hypothèses, DAG, ce que chaque test peut voir ; notebook 02 |
| 3 Estimation | `03_estimation/` | BSTS par campagne, SC / SC augmenté, DiD, DML intensité promo ; notebooks 03, 05 |
| 4 Backtesting | `04_backtesting/` | placebos, vérité connue sur panel réel, couverture par estimateur, séisme ; notebooks 04, 06 |
| 5 Pipeline | `05_pipeline/` | Dagster local, MLflow, manifestes de données, calendrier optimisé |
| 6 Monitoring | `06_monitoring/` | signaux de dérive, placebos périodiques, réconciliation de marge ; notebook 07 |

Le README racine porte le tableau « ce que je corrige par rapport au cahier des charges » avec les chiffres finaux, et un recul sincère sur ce que la méthode n'a pas pu mesurer.
