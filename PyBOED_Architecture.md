# PyBOED — Document d'architecture

> **Bibliothèque Python pour le Bayesian Optimal Experimental Design**  
> AIRSEA · Inria Grenoble · Université Grenoble Alpes  
> JAX-first · Python 3.11+ · GitHub + GRICAD

---

## À propos de ce document

Ce document est vivant. Il est fait pour être annoté, modifié, et évoluer avec la thèse.
Il ne prétend pas tout figer — certaines parties sont stables, d'autres sont encore ouvertes.
La convention utilisée dans tout le document :

| Symbole | Signification |
|---------|---------------|
| ✅ | Stable — à implémenter en premier |
| 🔶 | Interface définie, implémentation différée |
| ❓ | Encore ouvert — dossier vide + README |

---

## Table des matières

1. [Le problème qu'on résout](#1-le-problème-quon-résout)
2. [Cas test de référence — Burgers 1D](#2-cas-test-de-référence--burgers-1d)
3. [Décisions techniques fondamentales](#3-décisions-techniques-fondamentales)
4. [Structure du projet](#4-structure-du-projet)
5. [core/ — les interfaces abstraites](#5-core--les-interfaces-abstraites)
6. [criteria/ et estimators/](#6-criteria-et-estimators)
7. [bounds/ — la contribution originale](#7-bounds--la-contribution-originale)
8. [linalg/ et priors/](#8-linalg-et-priors)
9. [optim/](#9-optim)
10. [surrogates/](#10-surrogates)
11. [viz/, io/, metrics/](#11-viz-io-metrics)
12. [Tests, docs, cluster](#12-tests-docs-cluster)
13. [Feuille de route](#13-feuille-de-route)
14. [Haute dimension, HPC et algèbre linéaire randomisée](#14-haute-dimension-hpc-et-algèbre-linéaire-randomisée)
15. [Sources — bibliographie complète](#15-sources--bibliographie-complète)
16. [Questions à poser aux directeurs](#16-questions-à-poser-aux-directeurs--version-complète)
17. [Environnement, outillage et workflow](#17-environnement-outillage-et-workflow)

---

## 1. Le problème qu'on résout

On a un modèle forward `G`, un paramètre inconnu `θ`, et on fait des observations bruitées :

```
y = G(θ) + ε,    ε ~ N(0, Γ_noise),    θ ~ N(0, Γ_prior)
```

La question BOED : **où placer mes capteurs** (le design `ξ`) pour apprendre le maximum sur `θ` ?

On répond en maximisant l'**Expected Information Gain (EIG)** :

```
EIG(ξ) = H(θ) − E_y[ H(θ | y, ξ) ]
```

C'est l'information moyenne gagnée sur `θ` après avoir observé `y` avec le design `ξ`.

### Cas goal-oriented

Parfois on ne veut pas reconstruire `θ` entier, mais seulement une quantité d'intérêt (QoI) :

```
y = G(θ, η) + ε,    θ = h(η) + ε_η
```

`η` est le design, `θ` est ce qu'on veut reconstruire, `h` est la fonction qui relie les deux.
Le cas standard est un cas particulier où `h` est l'identité.

### Ce qui est difficile

- `G` peut être une EDP coûteuse → on ne peut pas évaluer l'EIG exactement
- L'espace des designs est grand (grille EDP 1D ou 2D)
- En non-linéaire, plus de formule fermée → Monte Carlo ou approximations
- Sur cluster, il faut paralléliser les appels à `G`

---

## 2. Cas test de référence — Burgers 1D

Tout PyBOED est validé sur **Burgers 1D avec un paramètre de non-linéarité `λ`** :

```
∂ₜu + λ u ∂ₓu = ν ∂ₓₓu,    λ ∈ [0, 1]
```

Pourquoi ce cas test est idéal :

| λ | Comportement | Ce que ça teste |
|---|---|---|
| 0 | Transport pur — linéaire | EIG exact, bornes confondues |
| 0.2 | Légèrement non-linéaire | Petit gap, première certification |
| 1 | Burgers complet | Gap large, estimateurs MC nécessaires |

C'est un **paramètre de contrôle continu de la non-linéarité** — on trace exactement la dégradation des bornes en fonction de `λ`. C'est le fil rouge de tous les tutoriels.

---

## 3. Décisions techniques fondamentales

### Backend : JAX-first

**JAX** est le bon choix pour PyBOED. Les raisons concrètes :

- `jax.jit` — compilation JIT sur CPU et GPU sans changer le code
- `jax.grad` — différentiation automatique pour les surrogates et l'optimisation interne
- `jax.vmap` — vectorisation sur les candidats du greedy
- `jax.numpy` — remplace numpy sans friction
- Compatible `flax` / `equinox` si on ajoute des surrogates NN plus tard
- Sur GRICAD : même code, CPU ou GPU selon le job SLURM

**PyTorch** : uniquement pour les surrogates NN, isolé dans `surrogates/`, dépendance optionnelle.  
**C++** : via `pybind11` pour wrapper des codes EDP externes, isolé dans `bindings/`.

### State : immuable et pytree

JAX impose l'immuabilité dans les fonctions `jit`-ées. Tout l'état du greedy est un `NamedTuple` — c'est un pytree JAX valide, `jit`/`vmap`/`grad` le traversent sans friction.

```python
# ✅ correct — retourne un nouveau state
state = criterion.update_state(state, best_xi)

# ❌ interdit — pas de mutation en place dans JAX jit
state.cov_post = new_cov
```

### PRNG : clés JAX explicites

Pas de seed globale. Toutes les fonctions stochastiques prennent une clé en argument :

```python
key = jax.random.PRNGKey(seed)
key, subkey = jax.random.split(key)
samples = jax.random.normal(subkey, shape=(n,))
```

La clé initiale est sauvegardée dans le fichier HDF5 → reproductibilité garantie.

### Versioning : semver mode thèse

- `0.x` : API instable assumée — casser l'interface entre `0.1` et `0.2` est normal
- `1.0.0` : stabilisation à la soumission d'un article ou fin de thèse

---

## 4. Structure du projet

```
pyboed/
├── core/           ✅  Interfaces abstraites — aucun calcul numérique ici
├── criteria/       ✅  EIG, D-optimal, goal-oriented, reconstruction
├── estimators/     ✅  Analytique LG, NMC, VNMC, Laplace
├── bounds/         ✅  Certification EIG — contribution originale
├── linalg/         ✅  Rank-1 updates, rSVD, sparse, Woodbury
├── priors/         ✅  Matérn 1D  |  🔶 Kronecker 2D
├── optim/          ✅  Greedy séquentiel  |  🔶 batch reopt
├── surrogates/     🔶  NeuralSurrogate (rôle A) + decoder f (rôle B)
├── backends/       ✅  Dispatch JAX / NumPy / Numba
├── bindings/       🔶  pybind11 — codes EDP C++ externes
├── viz/            ✅  design, covariance, bornes, boxplots, animate
├── io/             ✅  HDF5, experiment multi-runs, seeds JAX
└── metrics/        ✅  EIG, RMSE, comparaisons OED vs random

tests/
├── unit/           ✅  Chaque brique isolée
├── integration/    ✅  Assemblages complets
├── regression/     ✅  Résultats de référence figés
└── performance/    ✅  Benchmarks de vitesse

docs/
└── tutorials/      ✅  Notebooks Burgers λ∈[0,1]

cluster/
├── job_template.slurm
└── launch_sweep.py

pyproject.toml
.pre-commit-config.yaml
README.md
CONTRIBUTING.md
```

---

## 5. core/ — les interfaces abstraites

> **Règle d'or** : si un fichier de `core/` contient du calcul numérique, c'est un bug d'architecture.

`core/` définit les **contrats** que tout le reste respecte. Aucune implémentation ici.

---

### ForwardModel — `core/forward_model.py` ✅

Représente `G : θ → y`. Ne sait pas s'il est linéaire ou non.

```
Méthode              Retour      Description
──────────────────────────────────────────────────────────────
__call__(theta)      array y     Évaluation G(θ)
jacobian(theta)      array J     ∂G/∂θ — différences finies ou AD
matvec(v)            array       H·v sans matérialiser H (matrix-free)
rmatvec(v)           array       Hᵀ·v — requis pour les solveurs linalg
```

**Sous-classes** (dans `criteria/` et `surrogates/`) :

- `LinearForwardModel` — H explicite (dense ou sparse Matrix Market) ou matrix-free.
  `jacobian()` retourne H directement. Accepte `scipy.sparse` et JAX BCSR.
- `NonlinearForwardModel` — `jacobian()` par différences finies. Slot prévu pour AD JAX.
- `GoalOrientedWrapper` — décore n'importe quel `ForwardModel`, ajoute `h(η)` et sa jacobienne.
- `NeuralSurrogate` — même interface, s'utilise de façon transparente à la place de tout `ForwardModel`.

---

### State — `core/state.py` ✅

Pytree JAX immuable. Accumule l'information au fil du greedy.
**L'optimiseur ne regarde jamais l'intérieur** — il délègue tout au critère.

```
Attribut             Type            Description
──────────────────────────────────────────────────────────────
selected_indices     tuple[int]      Capteurs déjà placés — immuable
criterion_value      float           Valeur courante pour logging
cov_post             array (n,n)     Covariance postérieure dense
U, d                 (n,r), (r,)     Représentation low-rank : Γ_post ≈ UDUᵀ
```

**Hiérarchie** :

```
LinearGaussianState(NamedTuple)
    cov_post dense par défaut
    (U, d) avec rank=r en option

NonlinearState(NamedTuple)
    jacobien local courant

GoalOrientedState(NamedTuple)
    inner_state : LinearGaussianState ou NonlinearState
    qoi_sensitivity
    h_jacobian
```

`GoalOrientedState` est un **wrapper** — il contient un `inner_state` de n'importe quel type.
Pas d'explosion combinatoire (LinearGoalOriented, NonlinearGoalOriented...).

---

### Criterion — `core/criterion.py` ✅

Sait calculer son score ET gérer son state.
Choisit automatiquement son estimateur selon le type de modèle reçu.

```
Méthode                      Retour    Description
──────────────────────────────────────────────────────────────
init_state(model, prior)     State     Initialise depuis le prior et le modèle
evaluate(state, xi)          float     Score pour un candidat xi — lecture seule
update_state(state, xi)      State     Nouveau state après ajout de xi
estimator_for(model)         Estimator Dispatch automatique
```

**Règles de dispatch** :

```
LinearForwardModel    →  analytical.py   formule exacte, pas de Monte Carlo
NonlinearForwardModel →  nmc.py          par défaut
                         laplace.py      via EIG(estimator='laplace')
```

> `update_state()` appelle `linalg/updates.py` pour les rank-1 updates.
> Le critère orchestre, `linalg/` calcule.

---

### DesignSpace — `core/design_space.py` ✅

```
Méthode / attribut      Description
──────────────────────────────────────────────────────────────
candidates              Toutes les positions candidates (grille EDP 1D)
remaining(selected)     Candidats non encore sélectionnés
select(indices)         Retourne un objet Design immuable
```

`DiscreteDesignSpace` est la seule implémentation pour l'instant. Le cas continu est différé.

---

### Optimizer — `core/optimizer.py` ✅

```
Méthode                              Retour   Description
──────────────────────────────────────────────────────────────
run(criterion, model, space, n)      Result   Boucle greedy principale
_step(state, space)                  xi,State Une étape : évalue + update
```

L'optimiseur est **intentionnellement stupide** : aucune logique numérique.

---

### Result ✅

```
Attribut      Type                  Description
──────────────────────────────────────────────────────────────
design        Design                Indices sélectionnés finaux
history       list[State]           State complet à chaque étape greedy
scores        array(n_steps,n_cand) Scores de tous les candidats à chaque étape
runtime       float                 Temps total en secondes
```

> `history` est la source de `animate_greedy()` et des boxplots multi-runs.
> Ne pas négliger cet objet — il conditionne toute la visualisation et l'io.

---

### Flux d'exécution type

```
prior + model
    │
    ▼
criterion.init_state()          ← initialise Γ_post = Γ_prior
    │
    ▼
pour k = 1 … n_sensors :
    │
    ├── pour chaque xi candidat :
    │       criterion.evaluate(state, xi)    ← rank-1 update cheap
    │
    ├── best_xi = argmax(scores)
    │
    └── state = criterion.update_state(state, best_xi)
    │
    ▼
Result(design, history, scores)
    │
    ├── io.save_run()
    └── viz.animate_greedy()
```

---

## 6. criteria/ et estimators/

### EIG — `criteria/eig.py` ✅

Critère principal : `I(θ; y | ξ) = H(θ) − H(θ|y)`.

En linéaire gaussien, formule exacte :

```
EIG(ξ) = ½ log det(I + H Γ_prior Hᵀ Γ_noise⁻¹)
```

En non-linéaire : estimateurs Monte Carlo (voir ci-dessous).

### D-optimalité — `criteria/doptimal.py` ✅

```
D(ξ) = log det(Γ_post(ξ)⁻¹) = −log det(Γ_post(ξ))
```

Équivalent à EIG dans le cas LG. Utile pour comparaison.

### Goal-oriented — `criteria/goal_oriented.py` 🔶

Wrapper sur n'importe quel critère pour le cas `θ = h(η) + ε_η`.
Injecte la jacobienne de `h` dans le calcul interne.

### Critère de reconstruction — `criteria/reconstruction.py` 🔶

```
min E‖G(θ) − f(Y)‖²
```

`f : Y → Ĝ(θ)` est un décodeur appris (dans `surrogates/decoder.py`).
Lien avec l'information de Fisher :

```
Cov(G(θ)|Y) ≥ E‖G(θ) − f(Y)‖²
```

Égalité quand `f` est l'estimateur de Bayes optimal.
Ce n'est **pas** un surrogate de `G` — c'est un critère d'optimalité en soi.
Dans le cas goal-oriented, `G(θ)` est remplacé par le QoI `h(η)`.

---

### Estimateurs

| Fichier | Méthode | Quand l'utiliser |
|---|---|---|
| `analytical.py` ✅ | Formule exacte | Linéaire gaussien uniquement |
| `nmc.py` ✅ | Nested Monte Carlo | Non-linéaire — borne basse biaisée |
| `vnmc.py` ✅ | Variational NMC | Non-linéaire — borne haute |
| `laplace.py` ✅ | Approximation Laplace | Fallback rapide, override explicite |

> NMC et VNMC utilisés **en tandem** encadrent l'EIG — connexion directe avec `bounds/`.

---

## 7. bounds/ — la contribution originale

Au lieu de calculer `EIG(ξ)` exactement (coûteux), on certifie un intervalle :

```
EIG_low(ξ)  ≤  EIG(ξ)  ≤  EIG_up(ξ)
```

### Pourquoi c'est important

- **Linéaire gaussien** : les bornes sont confondues (formule exacte). Gap = 0.
- **Non-linéaire** : le gap s'élargit. Sa largeur est une mesure de difficulté du problème.
- **Goal-oriented** : le gap vient de deux sources — la non-linéarité de `G` **et** celle de `h`.

### Décomposition du gap (cas goal-oriented)

```
gap_total = gap_G + gap_h + interaction
             ↑        ↑
        non-lin G   non-lin h
```

C'est une contribution originale : même si `G` est linéaire, `h` non-linéaire crée un gap.

---

### Interface commune — `bounds/base.py` ✅

```python
bounder = LinearizationBound(model, prior)
result  = bounder.compute(design)

result.lower          # borne basse
result.upper          # borne haute
result.gap            # upper - lower
result.nonlinearity   # mesure de dégradation
result.is_certified   # gap < tolerance ?
```

---

### Fichiers du module

| Fichier | Contenu |
|---|---|
| `base.py` ✅ | Interface `EIGBound` + `BoundResult` |
| `linearization.py` ✅ | Borne via linéarisation locale (Laplace) |
| `mc_bounds.py` ✅ | NMC lower + VNMC upper formalisés comme bornes certifiées |
| `nonlinearity.py` ✅ | `BurgersNonlinearityMeasure(λ)` + `JacobianVariation` générique |
| `goal_oriented.py` ✅ | `GoalOrientedBound` — wrapper + décomposition gap_G / gap_h |

---

### Intégration avec le greedy — mode certifié

```python
for xi in candidates:
    bound = bounder.compute(state, xi)

    if bound.gap < tolerance:
        score = bound.lower        # certifié suffisamment précis, pas besoin de plus
    else:
        score = estimator.compute(state, xi)   # fallback coûteux
```

C'est un **critère d'arrêt adaptatif** : l'optimiseur s'adapte à la difficulté locale.

---

### Tableau de validation — Burgers 1D

| λ | h (QoI) | Gap attendu | Tutorial |
|---|---|---|---|
| 0 | linéaire | 0 — bornes confondues | `01_linear_gaussian.ipynb` |
| 0 | non-linéaire | gap pur h | `02_burgers_weak.ipynb` |
| 1 | linéaire | gap pur G | `03_burgers_strong.ipynb` |
| 1 | non-linéaire | gap total décomposé | `04_goal_oriented.ipynb` |

---

## 8. linalg/ et priors/

### Rank-1 updates — `linalg/updates.py` ✅

Cœur computationnel du greedy incrémental.
À chaque étape, mise à jour de Woodbury au lieu de recalculer depuis zéro :

```
Γ_post^(k+1) = Γ_post^(k) − Γ_post^(k) hᵢᵀ (σ² + hᵢ Γ_post^(k) hᵢᵀ)⁻¹ hᵢ Γ_post^(k)
```

En représentation low-rank `(U, d)`, l'update opère directement sur les facteurs.

### Algèbre linéaire randomisée — `linalg/randomized.py` ✅

- `randomized_svd(A, rank, n_oversampling)` — approximation low-rank de Γ_prior et Γ_post
- `nystrom(A, rank)` — approximation de Nyström pour matrices SPD
- `randomized_range_finder(A, rank)` — base orthonormée de l'image de A

### Matrices sparses — `linalg/sparse.py` ✅

- `load_matrix_market(path)` → `scipy.sparse` ou JAX BCSR selon le backend
- `sparse_matvec(A, v)` — produit sparse-vecteur compatible `jit`

### Prior de Matérn — `priors/matern.py` ✅

Les priors de Matérn ont un spectre à décroissance rapide → idéal pour low-rank.

```
Méthode                   Description
──────────────────────────────────────────────────────────────
__init__(grid,nu,ℓ,rank)  rank=None → dense, rank=r → low-rank
sample(n_samples, key)    Échantillonnage JAX-compatible
low_rank_approx(rank)     Retourne (U, d) tels que Γ_prior ≈ UDUᵀ
matvec(v)                 Γ_prior · v — matrix-free
log_prob(theta)           Log-densité en θ
```

> **Kronecker 2D** 🔶 : pour les grilles tensorielles, `Γ_prior = Γ_x ⊗ Γ_y`.
> Les produits matrice-vecteur passent de O(n²) à O(n√n). Différé dans `priors/matern_2d.py`.

---

## 9. optim/

Trois niveaux distincts, même interface de base :

```python
Optimizer.run(objective, space, budget) → Result
```

L'objectif peut être un critère BOED, une loss de reconstruction, ou une log-vraisemblance.
L'optimiseur ne sait pas lequel.

### Niveau 1 — design — `optim/design.py` ✅

Trouve les k positions qui maximisent le critère.

- `GreedyOptimizer` — greedy séquentiel avec rank-1 updates incrémentaux
- `GreedyBatchReopt` 🔶 — réoptimise l'ensemble du design courant à chaque étape.
  Plus coûteux (facteur k), corrige les erreurs des étapes précédentes.
  *(Suggestion d'une conférence — prévu mais non prioritaire)*

### Niveau 2 — interne — `optim/inner.py` 🔶

Optimisation interne au critère de reconstruction :

```
min_{params(f)}  E‖G(θ) − f(Y)‖²
```

Utilise `jax.grad` sur les paramètres du réseau `f`.
La question de l'apprentissage conjoint (design + décodeur optimisés ensemble) est encore ouverte.

### Niveau 3 — hyperparamètres — `optim/hyperparam.py` ✅

Recherche sur grille pour :
- Longueur de corrélation et ν du prior Matérn
- Rang de la réduction de dimension
- Nombre de samples NMC

---

## 10. surrogates/

Deux rôles distincts — ne pas confondre.

### Rôle A — `surrogates/neural_forward.py` 🔶

`NeuralSurrogate` apprend `Ĝ ≈ G` pour remplacer des appels coûteux.

**Implémente exactement la même interface que `ForwardModel`** — le reste du code ne sait
pas qu'il y a un réseau. S'utilise de façon transparente partout où un `ForwardModel` est attendu.

### Rôle B — `surrogates/decoder.py` 🔶

`Decoder` apprend `f : Y → Ĝ(θ)`, le décodeur du critère de reconstruction.

Ce n'est **pas** un surrogate de `G`.
C'est la fonction apprise dans `min E‖G(θ) − f(Y)‖²`, pilotée par `optim/inner.py`.
Dans le cas goal-oriented, `G(θ)` est remplacé par `h(η)`.

---

## 11. viz/, io/, metrics/

### viz/ ✅

Trois contextes d'usage → un style adapté :

| Contexte | Outil | Format |
|---|---|---|
| Exploration quotidienne | matplotlib interactif | inline notebook |
| Figures pour papiers | matplotlib + `style.mplstyle` | PDF vectoriel, LaTeX |
| Présentations / posters | plotly ou matplotlib | HTML interactif ou PDF |

**`viz/design.py`**
- `plot_design_1d(grid, selected)` — positions des capteurs sur la grille
- `plot_greedy_history(scores)` — score du meilleur candidat à chaque étape
- `animate_greedy(run_path, quantity='variance')` — animation des étapes greedy.
  Charge `(U, d)` à la volée, reconstruit `diag(UDUᵀ)` sans matérialiser la matrice dense.
  Source principale pour les présentations et posters.

**`viz/covariance.py`**
- `plot_prior_cov(prior)`, `plot_posterior_cov(state)`
- `plot_variance_reduction(before, after)`
- `plot_spectrum(U, d)` — décroissance du spectre de la représentation low-rank

**`viz/bounds.py`**
- `plot_eig_bounds(designs, lower, upper)` — intervalle sur le landscape EIG
- `plot_gap_vs_nonlinearity(results)` — dégradation du gap en fonction de λ (Burgers)
- `plot_certification_map(grid, bounder)` — quelles régions sont certifiées ?

**`viz/validation.py`**
- `plot_boxplot_metric(experiment, metric)` — distribution d'une métrique sur n_runs
- `plot_design_stability(experiment)` — heatmap de fréquence de sélection.
  Si le capteur k est choisi dans 48 runs sur 50, le design est robuste.
- `plot_oed_vs_random(designs)` — comparaison OED vs baselines

**`viz/style.mplstyle`** — un fichier de style appliqué une seule fois.
Palette fixe, police LaTeX, tailles papier (colonne simple / double) et poster.

---

### io/ ✅

**`io/hdf5.py`** — trois niveaux de verbosité :

```
light   → scores + design final         ~Ko    notebooks exploratoires
normal  → + low-rank (U,d) par étape    ~Mo    usage par défaut
full    → tout pour rejouer le run      ~Go    cluster uniquement
```

Structure du fichier HDF5 :

```
run_20240615_greedy_lg_1d.h5
├── metadata/    timestamp · version · config.json · seed JAX
├── design/      selected_indices · greedy_history/scores_step_k
├── state/       cov_prior · U/d par étape · criterion_values
└── model/       H (si LG) · grid EDP
```

**`io/experiment.py`** — pour les expériences répétées :

```python
exp = load_experiment("runs/greedy_lg_1d/")    # charge tous les run_*.h5
exp.metric_distribution("rmse")                # shape (n_runs,) → boxplot
```

---

### metrics/ ✅

- `information.py` — EIG exact (LG), estimé (NMC/VNMC), D-opt, A-opt
- `reconstruction.py` — RMSE, erreur relative, couverture intervalles crédibles
- `comparison.py` — fonction principale :

```python
benchmark_design(model, prior, designs_dict, n_trials=100, key=...)
# designs_dict = {"OED greedy": sel_oed, "aléatoire": sel_rand, "uniforme": sel_unif}
# → DataFrame avec RMSE, EIG, intervalles — prêt pour plot_oed_vs_random()
```

---

## 12. Tests, docs, cluster

### Quatre niveaux de tests

| Niveau | Dossier | Rôle |
|---|---|---|
| Unit | `tests/unit/` | Chaque brique isolée — le rank-1 update est-il correct ? |
| Intégration | `tests/integration/` | Assemblages — un greedy LG 1D complet donne le bon design ? |
| Régression | `tests/regression/` | Résultats de référence figés — détecte les régressions numériques |
| Performance | `tests/performance/` | Benchmarks — détecte les régressions de vitesse |

> Un tutoriel qui ne tourne pas est un bug. Les notebooks `docs/tutorials/` font office
> de tests d'intégration de bout en bout.

### Tutoriels — fil rouge Burgers

```
01_linear_gaussian.ipynb         λ=0 — cas LG exact, bornes confondues
02_burgers_weak_nonlinear.ipynb  λ=0.2 — gap petit, première certification
03_burgers_strong_nonlinear.ipynb λ=1 — Burgers complet, gap large
04_goal_oriented_burgers.ipynb   cas goal-oriented, décomposition gap
05_certification_landscape.ipynb gap en fonction de λ et du design
06_sparse_matrix_market.ipynb    matrices réelles issues de problèmes externes
07_experiment_boxplots.ipynb     répétitions, robustesse, visualisation statistique
```

### Infrastructure

```
.github/workflows/ci.yml      ruff + pytest + couverture, à chaque push
.github/workflows/docs.yml    build Sphinx automatique
.pre-commit-config.yaml       ruff + mypy avant chaque commit

cluster/job_template.slurm    template GRICAD avec modules, GPU/CPU, mémoire
cluster/launch_sweep.py       sweep de paramètres (λ, n_sensors, rank) en parallèle
```

### Dépendances — `pyproject.toml`

```
Obligatoires : jax · jaxlib · numpy · scipy · h5py · matplotlib
Dev          : ruff · pytest · pytest-cov · sphinx · myst-parser
Optionnelles : pip install pyboed[surrogates]  →  torch · flax ou equinox
```

---

## 13. Feuille de route

### Phase 1 — noyau LG ✅ *commencer ici*

- `core/` complet (interfaces abstraites)
- `linalg/` — rank-1 updates, rSVD, sparse
- `priors/matern.py` — 1D grille régulière
- `criteria/eig.py` — analytique LG
- `optim/design.py` — GreedyOptimizer
- `io/hdf5.py` — verbosity light/normal
- `viz/design.py` — plot_design_1d, plot_greedy_history

### Phase 2 — bornes et certification ✅

- `bounds/` complet
- `estimators/nmc.py` + `estimators/vnmc.py`
- `viz/bounds.py`
- Tutoriels 01 à 05 (Burgers λ variable)

### Phase 3 — non-linéaire et goal-oriented 🔶

- `estimators/laplace.py`
- `criteria/goal_oriented.py`
- `bounds/goal_oriented.py`
- Tutoriels 04 et 05 complets

### Phase 4 — surrogates 🔶

- `surrogates/neural_forward.py` (rôle A)
- `surrogates/decoder.py` (rôle B)
- `optim/inner.py`
- PyTorch en dépendance optionnelle

### Phase 5 — C++ et 2D ❓

- `bindings/` — pybind11 quand un code EDP externe en a besoin
- `priors/matern_2d.py` — Kronecker pour grilles tensorielles

---

### Questions encore ouvertes

- **Apprentissage conjoint** : `f` (décodeur) et design optimisés simultanément dans
  `optim/inner.py` — dépend des expériences numériques à venir.
- **Grilles irrégulières** : `DesignSpace` sur maillage EDP non-cartésien — différé.
- **Formalisation des bornes MC** : NMC/VNMC comme encadrement certifié — en cours de réflexion.
- **Optimiseur non-greedy** : gradient-based ou stochastique sur design continu — différé.

---

*Document généré lors de la discussion d'architecture — à faire évoluer au fil de la thèse.*

---

## 14. Haute dimension, HPC et algèbre linéaire randomisée

> Les deux tiers de la thèse portent sur le passage à très haute dimension.
> Cette section est le cœur computationnel du projet — pas un détail d'implémentation.

### 14.1 Ce qui change en haute dimension

En basse dimension, les choix d'implémentation sont flexibles.
En haute dimension, certaines contraintes deviennent absolues :

| Quantité | Basse dimension | Haute dimension |
|---|---|---|
| `Γ_post` (n×n) | Dense acceptable | **Interdit** — low-rank obligatoire |
| `log det(A)` | Cholesky exact | **Estimateur stochastique** |
| Rank-1 updates | Woodbury sur dense | **Woodbury sur facteurs (U, d)** |
| Prior Matérn | Matrice discrétisée | **Opérateur + structure Kronecker** |
| EIG | Formule directe | **Approximation + bornes** |
| Greedy | Séquentiel suffit | **Stabilité numérique critique** |

La représentation low-rank `(U, d)` n'est plus une option — c'est **la seule façon de fonctionner**.
Tout le reste du design découle de cette contrainte.

---

### 14.2 linalg/ — le vrai cœur de la bibliothèque

En haute dimension, `linalg/` n'est plus un module utilitaire.
C'est le module le plus important de PyBOED. Il faut le subdiviser :

```
linalg/
├── updates.py       ✅  Rank-1 updates en représentation low-rank — stabilité numérique
├── randomized.py    ✅  rSVD, Nyström, randomized range finder
├── logdet.py        ✅  Estimateurs stochastiques du log-déterminant
├── operators.py     ✅  Opérateurs matrix-free — matvec sans matérialiser
├── sparse.py        ✅  Sparse Matrix Market, JAX BCSR
└── distributed.py   ❓  Parallélisme multi-GPU / multi-nœud
```

#### linalg/updates.py — rank-1 updates en low-rank ✅

En haute dimension, on ne stocke jamais `Γ_post` dense.
On maintient la factorisation `Γ_post ≈ U D Uᵀ` et on met à jour les facteurs directement.

Après ajout du capteur `xᵢ` (vecteur `hᵢ` dans l'espace des observations) :

```
# Woodbury sur les facteurs low-rank
# Γ_post^(k+1) = Γ_post^(k) − Γ_post^(k) hᵢᵀ (σ² + hᵢ Γ_post^(k) hᵢᵀ)⁻¹ hᵢ Γ_post^(k)
#
# En low-rank (U, d) :
# 1. w = Uᵀ hᵢ                          O(n·r)
# 2. s = σ² + hᵢᵀ U D Uᵀ hᵢ             O(r)
# 3. U_new, d_new = mise à jour rang-1   O(n·r)
```

**Question ouverte** : la stabilité numérique des updates successifs en grande dimension.
Après k updates, les facteurs `(U, d)` peuvent se dégrader — il faut une stratégie de
reorthogonalisation périodique. C'est une vraie question de recherche.

#### linalg/logdet.py — log-déterminant stochastique ✅

En haute dimension, `log det(I + H Γ_prior Hᵀ Γ_noise⁻¹)` ne se calcule pas par Cholesky.
Trois approches à implémenter et comparer :

**Estimateur de Hutchinson** (simple, point de départ) :

```
log det(A) ≈ tr(log A) ≈ (1/m) Σ zᵢᵀ log(A) zᵢ,   zᵢ ~ N(0, I)
```

**Lanczos stochastique** (plus précis, plus coûteux) :

```
# Algorithme de Lanczos sur A avec vecteur aléatoire
# Approximation de log det via la forme quadratique de Gauss-Radau
# Compatible avec les rank-1 updates — c'est l'avantage clé
```

**Chebyshev** (alternative, bon pour les matrices creuses) :

```
# Approximation polynomiale de log(x) sur le spectre de A
# Nombre de termes contrôle précision/coût
```

> La compatibilité avec les rank-1 updates du greedy est le critère principal
> pour choisir entre ces approches — pas la précision absolue.

#### linalg/operators.py — opérateurs matrix-free ✅

En très haute dimension, on ne matérialise jamais `H` ou `Γ_prior`.
On travaille uniquement avec des opérateurs `v → A·v`.

```python
class LinearOperator:
    def matvec(self, v): ...     # A·v
    def rmatvec(self, v): ...    # Aᵀ·v
    def matmat(self, V): ...     # A·V  (plusieurs colonnes)
    def shape(self): ...         # (m, n)

# Compositions fréquentes :
# HΓH ᵀ = H @ Γ_prior @ Hᵀ  — jamais matérialisé
# (I + HΓHᵀ Γ_noise⁻¹)v     — résolu par Lanczos ou CG
```

JAX supporte cela nativement via `jax.scipy.sparse.linalg` (CG, GMRES)
et `jax.linear_util` pour les opérateurs fonctionnels.

#### linalg/randomized.py — algèbre linéaire randomisée ✅

Trois algorithmes fondamentaux, tous issus de Halko-Martinsson-Tropp (2011) :

**Randomized SVD** — approximation low-rank de A :

```
# 1. Ω = matrice aléatoire gaussienne (n × (r + p))   p = oversampling (~10)
# 2. Y = A Ω                                           matvecs
# 3. Q = orth(Y)                                       QR
# 4. B = Qᵀ A                                         matvecs
# 5. SVD de B (petit)  →  U, Σ, Vᵀ
# Coût : (r + p) matvecs au lieu de n
```

**Nyström** — pour les matrices SPD (cas de Γ_prior, Γ_post) :

```
# Plus efficace que rSVD pour les matrices SPD
# Exploite la symétrie définie positive
# Ω = matrice aléatoire (n × r)
# C = A Ω,  W = Ωᵀ C
# A ≈ C W⁻¹ Cᵀ  (factorisation Nyström)
```

**Randomized range finder** — base orthonormée de l'image de A :

```
# Avec power iteration pour améliorer la précision sur les spectres à décroissance lente
# Q = orth((AAᵀ)^q A Ω),  q = 1 ou 2 en pratique
```

---

### 14.3 priors/ — opérateurs différentiels et structure Kronecker

En haute dimension, le prior Matérn ne se stocke plus comme une matrice discrétisée.

#### Matérn comme solution d'une EDPS

La clé : un champ gaussien de Matérn avec paramètre `ν = p + d/2` est la solution stationnaire
de l'EDPS `(κ² − Δ)^(p+d/2) τ = W` où `W` est un bruit blanc.

Ça permet de représenter `Γ_prior` comme **un opérateur différentiel**, pas une matrice :

```python
class MaternOperator:
    # Γ_prior v  →  résoudre (κ² − Δ)^s u = v par méthodes spectrales ou FEM
    # Jamais de matrice dense — uniquement des matvecs
    def matvec(self, v): ...
```

C'est exactement le cadre des **problèmes inverses en dimension infinie**
(Alexanderian et al. 2016 — voir sources section 15).

#### Structure Kronecker pour les grilles tensorielles ✅ → 🔶

Pour une grille 2D tensorielle `(x, y)` :

```
Γ_prior = Γ_x ⊗ Γ_y
```

Avantages :
- `Γ_prior · v` coûte `O(n_x · n_y · (n_x + n_y))` au lieu de `O((n_x · n_y)²)`
- Le rSVD de `Γ_prior` se fait séparément sur `Γ_x` et `Γ_y`
- Compatible avec les matvecs matrix-free

Cette structure se propage aux rank-1 updates si `H` a aussi une structure Kronecker
(capteurs indépendants dans chaque direction) — à vérifier sur le modèle océan.

---

### 14.4 Estimateurs du log-déterminant — détail des méthodes

Le log-déterminant apparaît dans l'EIG linéaire gaussien :

```
EIG(ξ) = ½ log det(I + H Γ_prior Hᵀ Γ_noise⁻¹)
```

En haute dimension, il faut l'estimer. Trois stratégies, avec leurs compromis :

#### Estimateur de Hutchinson

```
tr(log A) ≈ (1/m) Σᵢ zᵢᵀ f(A) zᵢ
```

- Simple à implémenter, `jit`-able facilement
- Convergence en `O(1/√m)` — lente
- Biais si le spectre de `A` est mal conditionné
- **Point de départ recommandé**

#### Lanczos stochastique (SLQ)

```
# Pour chaque vecteur test z :
# 1. Algorithme de Lanczos sur A avec z → tridiagonale T_m
# 2. log det(A) ≈ ‖z‖² eᵀ₁ log(T_m) e₁  (quadrature de Gauss)
```

- Plus précis que Hutchinson à budget de matvecs égal
- Compatible avec les rank-1 updates — les matvecs de Lanczos s'appliquent à la nouvelle matrice
- Référence : Ubaru, Chen, Saad (2017)
- **À implémenter en phase 2 de `linalg/`**

#### Déterminant exact via rSVD (cas particulier)

Si `H Γ_prior Hᵀ` est de rang `r` petit (ce qui arrive quand le nombre de capteurs est petit
devant la dimension) :

```
log det(I + H Γ_prior Hᵀ Γ_noise⁻¹) = Σᵢ log(1 + σᵢ²/σ_noise²)
```

où `σᵢ` sont les valeurs singulières de `Γ_noise^(-½) H Γ_prior^(½)`.
En pratique c'est le cas pendant les premières étapes du greedy.

---

### 14.5 Réduction de dimension — deux niveaux distincts

La réduction de dimension intervient à **deux endroits distincts** avec des justifications
mathématiques différentes.

#### Côté paramètres θ (espace prior) ✅

Le prior `Γ_prior` a un spectre à décroissance rapide (Matérn).
On travaille dans le sous-espace des `r` directions les plus informatives :

```
Γ_prior ≈ U_r D_r U_rᵀ,    U_r ∈ ℝ^{n×r},  r << n
```

**Justification** (Spantini et al. 2015) : les directions orthogonales à `U_r` ne sont pas
mises à jour par les données — le posterior = prior dans ces directions.
Le rang `r` optimal est déterminé par le spectre de l'opérateur
`Γ_prior^{½} Hᵀ Γ_noise^{-1} H Γ_prior^{½}` (valeurs propres de Fisher).

#### Côté observations y (espace design) ✅

Si le nombre de capteurs candidats est grand, on peut réduire l'espace de recherche.
Moins standard, plus spécifique au problème — à explorer selon les résultats numériques.

#### Interaction entre les deux réductions ❓

**Question ouverte centrale** : si on réduit les deux espaces, est-ce que les erreurs
s'accumulent ? Y a-t-il une borne sur l'erreur d'approximation de l'EIG en fonction
des deux rangs de troncature ?

C'est une question à poser aux directeurs — elle conditionne la validité des résultats HPC.

---

### 14.6 HPC — parallélisme et cluster GRICAD

#### Parallélisme de modèles (priorité 1)

Le cas le plus simple et le plus efficace : plusieurs évaluations de `G` en parallèle.

```python
# JAX vmap — vectorisation sur les candidats du greedy
# Évaluer G(θ, xᵢ) pour tous les xᵢ en parallèle sur GPU
scores = jax.vmap(lambda xi: criterion.evaluate(state, xi))(candidates)
```

#### Parallélisme multi-GPU (priorité 2)

```python
# JAX pmap — parallélisme de données sur plusieurs GPUs
# Pour NMC avec beaucoup de samples :
log_likelihoods = jax.pmap(compute_log_likelihood)(samples_sharded)
```

Référence : documentation JAX sur `pmap` et `jax.sharding` (API plus récente).

#### Parallélisme distribué MPI (priorité 3 — différé) ❓

Si le forward model `G` est un code EDP MPI (NEMO, MOM6, ou code interne),
le parallélisme se gère côté `bindings/` via `mpi4py` ou directement dans le code C++.
PyBOED orchestre, le code MPI s'exécute.

#### Structure d'un job GRICAD type

```bash
# cluster/job_template.slurm
#SBATCH --nodes=1
#SBATCH --gres=gpu:1              # ou gpu:4 pour multi-GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu

module load cuda/12.0
module load python/3.11

# Variables JAX importantes sur cluster
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # évite OOM JAX
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

python launch_experiment.py \
    --n_sensors 50 \
    --rank 100 \
    --lambda_burgers 0.5 \
    --verbosity full \
    --output_dir $SCRATCH/pyboed_runs/
```

#### Sweep de paramètres

```python
# cluster/launch_sweep.py
# Soumet une grille de jobs : λ × n_sensors × rank
params = {
    "lambda": [0.0, 0.2, 0.5, 1.0],
    "n_sensors": [10, 20, 50],
    "rank": [50, 100, 200],
}
# → 4 × 3 × 3 = 36 jobs SLURM indépendants
```

---

### 14.7 distributed.py — perspective long terme ❓

En très haute dimension sur plusieurs nœuds, les opérations linéaires doivent être distribuées.
Les options :

- **JAX sharding** (recommandé si on reste JAX) — distribue les tableaux sur plusieurs GPUs
  sans changer le code de calcul. API stable depuis JAX 0.4.
- **PETSc / SLEPc** (si le forward model est déjà en C++ MPI) — via `petsc4py`.
  Solveurs linéaires et problèmes aux valeurs propres distribués. Très mature.
- **mpi4jax** — MPI + JAX. Encore expérimental, à surveiller.

> Ne pas implémenter maintenant. Concevoir `linalg/operators.py` de façon à ce que
> le passage au distribué ne change pas l'interface publique.

---

## 15. Sources — bibliographie complète

### Théorie BOED

| Référence | Pourquoi lire |
|---|---|
| Chaloner & Verdinelli (1995) — *Bayesian Experimental Design: A Review*, Statistical Science | Point de départ historique |
| Ryan et al. (2016) — *Fully Bayesian Experimental Design*, Entropy | Premier NMC sérieux |
| Foster et al. (2019) — *Variational Bayesian Optimal Experimental Design*, NeurIPS | VNMC et les bornes |
| Foster et al. (2021) — *Deep Adaptive Design*, ICML | Amortized BOED |
| Kleinegesse & Gutmann (2020) — *Efficient Bayesian Experimental Design for Implicit Models* | Bornes EIG |

### BOED haute dimension et problèmes inverses

| Référence | Pourquoi lire |
|---|---|
| Alexanderian et al. (2016) — *A-optimal design for infinite-dimensional Bayesian linear inverse problems*, SIAM | **Exactement ton cadre** |
| Alexanderian (2021) — *Optimal experimental design for infinite-dimensional Bayesian inverse problems* | Review récente, très pertinente |
| Spantini et al. (2015) — *Optimal low-rank approximations of Bayesian linear inverse problems*, SIAM | **Fondamental pour la réduction** |
| Cui et al. (2014) — *Likelihood-informed dimension reduction for nonlinear inverse problems* | Cas non-linéaire |
| Wu & Chen (2023) — travaux récents BOED + dimension réduite | Cherche sur arXiv |

### Algèbre linéaire randomisée

| Référence | Pourquoi lire |
|---|---|
| Halko, Martinsson, Tropp (2011) — *Finding Structure with Randomness*, SIAM Review | **Référence principale — lire en entier** |
| Martinsson & Tropp (2020) — *Randomized Numerical Linear Algebra*, Acta Numerica | Version complète et récente |
| Tropp et al. (2017) — *Practical Sketching Algorithms for Low-Rank Matrix Approximation* | Aspects computationnels |
| Ubaru, Chen, Saad (2017) — *Fast Estimation of tr(f(A)) via Stochastic Lanczos Quadrature* | **Log-déterminant par Lanczos** |
| Hutchinson (1990) — *A stochastic estimator of the trace* | Estimateur de trace simple |
| Golub & Meurant — *Matrices, Moments and Quadrature* | Quadrature de Gauss pour log-det |

### Priors et opérateurs différentiels

| Référence | Pourquoi lire |
|---|---|
| Lindgren, Rue, Lindström (2011) — *An explicit link between Gaussian fields and GMRFs*, JRSS | Matérn comme solution d'EDPS |
| Stuart (2010) — *Inverse problems in a Bayesian setting*, Acta Numerica | Cadre dimension infinie |
| Dashti & Stuart (2017) — *The Bayesian Approach to Inverse Problems* | Review complète |

### HPC et JAX

| Référence | Pourquoi lire |
|---|---|
| Documentation JAX — jax.readthedocs.io | *Thinking in JAX*, *Pytrees*, *pmap*, *sharding* |
| Patrick Kidger — blog et Equinox | JAX fonctionnel, bonnes pratiques |
| Documentation PETSc / SLEPc | Si passage à MPI distribué |

### Implémentations de référence à lire

| Dépôt | Ce qu'on y apprend |
|---|---|
| `pyro-ppl/boed` (GitHub) | Choix d'architecture BOED en PyTorch |
| `scikit-learn` source | Interfaces abstraites bien pensées |
| `jax` source | Comment JAX structure ses opérateurs |
| `FEniCS` / `dolfinx` | Opérateurs différentiels en Python |

### Numérique pour Burgers 1D

| Référence | Pourquoi lire |
|---|---|
| LeVeque — *Finite Volume Methods for Hyperbolic Problems*, Cambridge | Implémentation numérique Burgers |
| Hesthaven & Warburton — *Nodal Discontinuous Galerkin Methods* | Si tu passes au DG |
| Cherche "Burgers equation data assimilation AIRSEA" sur arXiv | Papiers de l'équipe sur ce cas |

---

## 16. Questions à poser aux directeurs — version complète

### Sur les bornes EIG

- Quelles sont les hypothèses minimales sur `G` pour que la borne de linéarisation soit rigoureuse ? Lipschitz sur le jacobien ?
- Le gap observé sur Burgers avec `λ` variable — peut-on le borner analytiquement en fonction de `λ` ?
- NMC donne une borne basse biaisée, VNMC une borne haute — ce biais est-il quantifiable dans le cadre gaussien ?
- La décomposition `gap_G + gap_h + interaction` dans le cas goal-oriented — a-t-elle un sens formel ?

### Sur le critère de reconstruction

- `Cov(G(θ)|Y) ≥ E‖G(θ) − f(Y)‖²` — minimiser ce critère sur le design donne-t-il le même optimum que maximiser l'EIG ?
- `f` optimal a-t-il une forme analytique dans le cas gaussien non-linéaire ?
- Remplacer `G(θ)` par `h(η)` dans le cas goal-oriented — justification théorique ou heuristique ?

### Sur la haute dimension

- Quel est le rang effectif de `Γ_post` après `k` capteurs, en fonction du prior Matérn et du forward model ? Peut-on le borner ?
- Si on réduit les deux espaces (paramètres et observations), les erreurs s'accumulent-elles ? Y a-t-il une borne sur l'erreur d'approximation de l'EIG ?
- La structure Kronecker de `Γ_prior` se propage-t-elle aux rank-1 updates si `H` n'a pas cette structure ?
- Pour le log-déterminant en grande dimension — Lanczos stochastique ou Hutchinson ? Critère de choix dans notre cadre ?

### Sur le design expérimental

- Y a-t-il des garanties de sous-modularité du critère dans notre cadre ? Ça conditionne la qualité de l'approximation greedy.
- La stabilité numérique des rank-1 updates successifs en grande dimension — est-ce un problème connu ? Stratégies de reorthogonalisation ?
- Comment choisir le nombre de capteurs `k` ? Critère d'arrêt lié aux bornes ?

### Sur la validation

- Burgers 1D suffit-il pour un premier papier, ou il faut un cas 2D ?
- Quelle métrique est la plus convaincante pour montrer l'utilité des bornes — largeur du gap, corrélation avec l'erreur vraie, gain computationnel ?
- Comment comparer rigoureusement OED vs design aléatoire — quelle métrique, combien de runs ?


---

## 17. Environnement, outillage et workflow

> Cette section couvre tout ce qui est autour du code — gestion des dépendances,
> qualité du code, documentation, et le cycle local → Git → GRICAD.

### 17.1 Pixi — gestion de l'environnement

**Pourquoi pixi** : reproductibilité exacte entre local et GRICAD via un lockfile,
gestion propre des environnements multiples (CPU, GPU, surrogates),
et remplacement de la chaîne conda + pip + environment.yml par un seul fichier.

**Principe** : tout vit dans `pyproject.toml` via les sections `[tool.pixi.*]`.
Le `pixi.lock` est versionné dans Git — il garantit que GRICAD installe exactement
les mêmes versions qu'en local.

```toml
# pyproject.toml — sections pixi

[tool.pixi.workspace]
channels  = ["conda-forge"]
platforms = ["linux-64"]

[tool.pixi.dependencies]
python     = "3.11.*"
jax        = ">=0.4.20"
scipy      = ">=1.11"
numpy      = ">=1.26"
h5py       = ">=3.10"
matplotlib = ">=3.8"

[dependency-groups]
test       = ["pytest", "pytest-cov"]
dev        = ["ruff", "sphinx", "myst-parser", "sphinx-autoapi", "furo"]
surrogates = ["pytorch>=2.1"]

[tool.pixi.environments]
default    = { solve-group = "default" }
test       = { features = ["test"],        solve-group = "default" }
dev        = { features = ["test", "dev"], solve-group = "default" }
surrogates = { features = ["surrogates"],  solve-group = "default" }

[tool.pixi.tasks]
test       = "pytest tests/unit"
test-all   = "pytest tests/"
lint       = "ruff check pyboed/"
format     = "ruff format pyboed/"
docs       = "sphinx-build docs/ docs/_build"
```

Le `solve-group = "default"` garantit que `pytest` a la même version dans `test` et `dev`.

---

### 17.2 Workflow local → Git → GRICAD

#### Setup initial (une seule fois)

```bash
# Installer pixi
curl -fsSL https://pixi.sh/install.sh | bash

# Initialiser le projet
pixi init pyboed --format pyproject
cd pyboed

# Ajouter les dépendances
pixi add jax scipy numpy h5py matplotlib
pixi add --pypi --feature test pytest pytest-cov
pixi add --pypi --feature dev ruff sphinx myst-parser furo
```

#### Développement quotidien en local

```bash
pixi run test      # pytest tests/unit/
pixi run test-all  # pytest tests/ — avant chaque push
pixi run lint      # ruff check — vérifie le style
pixi run format    # ruff format — formate automatiquement
pixi run docs      # génère docs/_build/index.html
```

#### Git

```bash
git init
git add pyproject.toml pixi.lock   # pixi.lock va dans Git — c'est voulu
git commit -m "init: setup pixi + dépendances de base"
git push
```

#### Sur GRICAD (nœud de login)

```bash
git clone https://github.com/toi/pyboed
cd pyboed
pixi install                   # lit pixi.lock exactement — reproductible
```

#### Sur GRICAD (nœud de calcul, sans internet)

```bash
# Dans le script SLURM :
export PATH="$HOME/.pixi/bin:$PATH"
pixi run --frozen python cluster/launch_experiment.py
# --frozen : utilise exactement le lockfile, ne tente pas de résoudre
```

---

### 17.3 .gitignore

```
# Environnement pixi — jamais dans Git
.pixi/

# Python
__pycache__/
*.pyc
*.pyo
.ruff_cache/
.pytest_cache/

# Documentation générée
docs/_build/

# Résultats — trop lourds pour Git
*.h5
runs/
outputs/

# Cluster
*.out
*.err
slurm-*.out
```

`pixi.lock` en revanche **va dans Git** — c'est lui qui garantit la reproductibilité.

---

### 17.4 Ruff — qualité du code

**Pourquoi ruff et pas black + flake8 + isort** : un seul outil remplace toute la chaîne,
10-100x plus rapide (écrit en Rust), et est devenu le standard de facto dans l'écosystème
Python scientifique — NumPy, SciPy, JAX l'ont tous adopté depuis 2023.

Config dans `pyproject.toml` :

```toml
[tool.ruff]
line-length    = 88
target-version = "py311"

[tool.ruff.lint]
select = [
    "E",   # erreurs de style (pycodestyle)
    "F",   # erreurs logiques (pyflakes) — imports inutilisés, variables non utilisées
    "I",   # ordre des imports (isort)
    "N",   # conventions de nommage (PEP8)
]
ignore = [
    "E501",  # longueur de ligne — géré par le formateur
]

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["F811"]  # redéfinitions acceptables dans les tests
```

**Pre-commit** — ruff tourne automatiquement avant chaque `git commit` :

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.0
    hooks:
      - id: ruff           # lint
      - id: ruff-format    # format
```

```bash
pixi add --feature dev pre-commit
pre-commit install   # à faire une seule fois après git init
```

Après ça, chaque `git commit` lance ruff automatiquement.
Si le code ne passe pas, le commit est bloqué — on ne pousse jamais de code mal formaté.

---

### 17.5 Sphinx — documentation

Sphinx génère un site web de documentation à partir des docstrings et de fichiers Markdown.
Standard universel en Python scientifique (NumPy, SciPy, JAX, scikit-learn).

**Extension MyST** : permet d'écrire en Markdown plutôt qu'en RST — beaucoup plus naturel.
**Thème Furo** : propre, moderne, celui adopté par JAX et beaucoup de projets récents.
**sphinx-autoapi** : génère la référence API automatiquement à partir des docstrings.

#### Structure docs/

```
docs/
├── conf.py           # configuration Sphinx
├── index.md          # page d'accueil
├── api.md            # référence API autogénérée par sphinx-autoapi
├── theory.md         # contexte mathématique
├── tutorials/
│   ├── 01_linear_gaussian.ipynb
│   ├── 02_burgers_weak.ipynb
│   └── ...
└── _build/           # généré — dans .gitignore
```

#### conf.py minimal

```python
project   = "PyBOED"
author    = "Ton nom"
release   = "0.1.0"

extensions = [
    "myst_parser",           # Markdown
    "autoapi.extension",     # API autogénérée
    "sphinx.ext.mathjax",    # équations LaTeX
    "sphinx.ext.viewcode",   # lien vers le code source
]

autoapi_dirs = ["../pyboed"]
html_theme   = "furo"
```

#### Format des docstrings — NumPy style

Tout PyBOED utilise le style NumPy (compatible sphinx-autoapi) :

```python
def randomized_svd(A, rank, n_oversampling=10, key=None):
    """
    Approximation SVD randomisée de A.

    Implémente Halko, Martinsson, Tropp (2011), section 4.1
    avec power iteration optionnelle.

    Parameters
    ----------
    A : LinearOperator ou array de forme (m, n)
        Matrice ou opérateur à approximer.
    rank : int
        Rang cible r, avec r << min(m, n).
    n_oversampling : int, optional
        Surééchantillonnage p. En pratique 5 à 10. Default : 10.
    key : jax.random.PRNGKey
        Clé PRNG JAX — obligatoire.

    Returns
    -------
    U : array de forme (m, r)
    s : array de forme (r,)
    Vt : array de forme (r, n)
        Tels que A ≈ U @ diag(s) @ Vt.

    References
    ----------
    Halko, Martinsson, Tropp (2011). Finding Structure with Randomness.
    SIAM Review, 53(2), 217–288.
    """
```

---

### 17.6 Récapitulatif de la chaîne d'outillage

```
pixi          gestion des environnements et dépendances
pixi.lock     reproductibilité exacte local ↔ GRICAD
ruff          linting + formatage (remplace black + flake8 + isort)
pre-commit    ruff automatique avant chaque git commit
pytest        tests (unit, intégration, régression, performance)
sphinx        documentation (MyST + furo + autoapi)
git           versioning — pyproject.toml et pixi.lock toujours commités
```

Commandes du quotidien :

```bash
pixi run format    # avant de committer
pixi run lint      # vérifie
pixi run test      # tests unitaires rapides
pixi run test-all  # avant chaque push
pixi run docs      # quand tu modifies la doc
```

