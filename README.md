# CBOED

## Computational Bayesian Optimal Experimental Design

A Python library for Bayesian optimal experimental design (BOED) in high-dimensional settings, with a focus on sensor placement for ocean models and large-scale inverse problems.

Developed as part of a PhD thesis at [AIRSEA](https://team.inria.fr/airsea/), Inria Grenoble — Université Grenoble Alpes.

---

## Overview

CBOED addresses the problem of finding optimal sensor configurations to maximize information gain about unknown parameters in geophysical models. Given a forward model $G$ and prior uncertainty on parameters $\theta$, the library finds the design $\xi$ that maximizes the **Expected Information Gain (EIG)**:

$$\text{EIG}(\xi) = H(\theta) - \mathbb{E}_y\left[H(\theta \mid y, \xi)\right]$$

The library is built to handle:

- **Linear and nonlinear forward models** — exact analytical solutions for the linear Gaussian case, Monte Carlo estimators for the nonlinear case
- **Goal-oriented design** — when only a quantity of interest $\theta = h(\eta)$ needs to be reconstructed
- **High-dimensional settings** — randomized linear algebra, low-rank representations, and matrix-free operators
- **EIG certification** — certified bounds $\text{EIG}_\text{low} \leq \text{EIG} \leq \text{EIG}_\text{up}$ instead of costly exact computation

---

## Status

Early development — `0.x` means the API is unstable and may change.

|Module|Status|
|---|---|
|`core/` — abstract interfaces|🔨 in progress|
|`linalg/` — randomized linear algebra|🔨 in progress|
|`criteria/` — EIG, D-optimal|🔨 in progress|
|`bounds/` — EIG certification|📋 planned|
|`priors/` — Matérn priors|📋 planned|
|`optim/` — greedy optimizer|📋 planned|
|`surrogates/` — neural surrogates|🔮 future|

---

## Installation

This project uses [pixi](https://pixi.sh) for environment management.

```bash
git clone https://github.com/mohamed495/CBOED.git
cd CBOED
pixi install
```

**Run the test suite:**

```bash
pixi run test-unit    # fast unit tests
pixi run test-all     # full test suite
```

**Check code quality:**

```bash
pixi run lint         # static analysis with ruff
pixi run format       # auto-format with ruff
```

**Build the documentation:**

```bash
pixi run docs
xdg-open docs/_build/index.html
```

---

## Reference case

The library is validated on **Burgers 1D** with a nonlinearity parameter $\lambda \in [0, 1]$:

$$\partial_t u + \lambda\, u\, \partial_x u = \nu\, \partial_{xx} u$$

- $\lambda = 0$ — linear transport, EIG bounds are exact (no gap)
- $\lambda = 1$ — full Burgers, large gap, Monte Carlo estimators required

This provides a continuous control of nonlinearity and serves as the reference benchmark for the entire library.

---

## Requirements

- Python 3.11+
- JAX (CPU by default, GPU via CUDA on cluster)
- See `pyproject.toml` for the full dependency list

Optional (neural surrogates):

```bash
pixi install --environment surrogates
```

### GPU (CUDA cluster, e.g. GRICAD)

No line in `cboed` forces `jax` onto the CPU (`jax_enable_x64` is the only
global config, in `cboed/__init__.py`) -- placing `jaxlib`'s CUDA build in the
environment is enough, no code change needed. `jax`/`jaxlib` 0.10.2 has a
matching `cuda129` build on `conda-forge`.

**Do this directly on the cluster, not on a machine without a GPU.** pixi
0.71.1 ties CUDA-capable virtual packages (`__cuda`) to the *platform*
(`linux-64`), not to an individual feature or environment: adding a `gpu`
feature with `system-requirements = { cuda = "12.9" }` to this repo's
`pyproject.toml` makes **every** environment on `linux-64` -- including
`test`, used for local dev -- require a CUDA-capable machine to install,
even if that environment never touches `jaxlib-cuda`. Tested and confirmed on
this dev machine (no GPU): `test` stopped installing as soon as that feature
was added, before any GPU-specific dependency was even requested.

Practical consequence: keep the GPU setup **local to the cluster checkout**
(uncommitted change, or a dedicated branch never merged back) rather than in
the shared `pyproject.toml` this laptop also uses:

```bash
# on the cluster, in your checkout
pixi add --feature gpu "jaxlib=0.10.2=cuda129*"
pixi run --environment gpu python -c "import jax; print(jax.devices())"
```

If `nvidia-smi` reports a different driver version than 12.9, adjust the
version constraint (`pixi search -c conda-forge "jaxlib=0.10.2=*cuda*"` lists
the available builds).

---

## Project structure

```text
cboed/
├── core/          Abstract interfaces — ForwardModel, State, Criterion, Optimizer
├── criteria/      EIG, D-optimal, goal-oriented, reconstruction
├── estimators/    Analytical (LG), NMC, VNMC, Laplace
├── bounds/        EIG certification — original contribution
├── linalg/        Rank-1 updates, randomized SVD, sparse, log-det estimators
├── priors/        Matérn priors, low-rank approximations
├── optim/         Greedy optimizer, hyperparameter search
├── surrogates/    Neural forward surrogates and decoders
├── viz/           Design, covariance, bounds, validation plots
├── io/            HDF5 storage, multi-run experiments
└── metrics/       EIG, RMSE, OED vs random comparisons
```

---

## Related work

- Alexanderian et al. (2016) — *A-optimal design for infinite-dimensional Bayesian linear inverse problems*, SIAM
- Halko, Martinsson, Tropp (2011) — *Finding Structure with Randomness*, SIAM Review
- Foster et al. (2019) — *Variational Bayesian Optimal Experimental Design*, NeurIPS
- Spantini et al. (2015) — *Optimal low-rank approximations of Bayesian linear inverse problems*, SIAM

---

## License

To be determined.

---

PhD thesis — AIRSEA, Inria Grenoble · Université Grenoble Alpes
