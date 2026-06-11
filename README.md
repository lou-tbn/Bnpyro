# Bnpyro

**Bnpyro** is a Python library that compiles higher-order probabilistic programs into exact Bayesian Networks (BNs). It uses [pyAgrum](https://agrum.gitlab.io/) for BN structure and inference, and is based on the formal framework of the **λ!-calculus** from:

> Faggian, Pautasso & Vanoni — *Higher-Order Bayesian Networks, Exactly* (POPL 2024)
> DOI: [10.1145/3632919](https://doi.org/10.1145/3632919)

Instead of approximate Monte Carlo inference (as in Pyro/Stan), Bnpyro compiles the program into a BN and performs **exact inference** via Variable Elimination.

---

## How it works

```python
from Bnpyro import BNppl
from distributions import Bernoulli

bn   = BNppl()
rain = bn.sample("rain", Bernoulli(0.2))
wet  = bn.sample("wet",  bn.where(rain, p_true=0.9, p_false=0.01))
bn.compile()
# BN compiled: 2 nodes, 1 arc, 6 CPT entries (0.000 MB)

bn.query("rain", evidence={"wet": True})
# {'False': 0.054, 'True': 0.946}
```

Each `bn.sample(...)` call becomes a BN node; each distribution becomes a CPT. `bn.compile()` wires everything together and `bn.query()` runs exact inference.

---

## Example: Rain → Wet

```python
bn   = BNppl()
rain = bn.sample("rain", Bernoulli(0.2))
wet  = bn.sample("wet",  bn.where(rain, 0.9, 0.01))
bn.compile()
bn.query("rain", evidence={"wet": True})
# {'False': 0.054, 'True': 0.946}
```

Posterior P(rain | wet=True) — exact inference, no sampling:

![Rain-Wet inference](img/rain_wet_inference.svg)

---

## Features

| Bnpyro construct | λ!-calculus | Description |
|---|---|---|
| `bn.sample("x", dist)` | `let x = sample_d` | Declare a random variable |
| `bn.sample("x", bn.where(pa, ...))` | `case⟨pa⟩` | Conditional Bernoulli CPT |
| `bn.sample("x", lambda p: dist, parents=[p])` | extension | Universal lambda (any parent type) |
| `bn.plate("f", dist)` / `f()` | `!t` / `der` | Freeze / instantiate a reusable distribution |
| `bn.recurse("x", fn, N)` | `fix f N` | Bounded recursion / Markov chains |
| Python HOF over `BNNode`/`BNThunk` | `λx.t` | Higher-order functions |
| `bn.query(target, evidence)` | VE | Exact posterior inference |

---

## Installation

```bash
pip install pyagrum scipy matplotlib
```

Clone or copy `Bnpyro.py` and `distributions.py` into your project.

---

## Quick Start

### Continuous variables

```python
from Bnpyro import BNppl
from distributions import Normal, Uniform

bn    = BNppl(n_bins=8)
mu    = bn.sample("mu",    Normal(0.0, 2.0))
sigma = bn.sample("sigma", Uniform(0.5, 2.0))
x     = bn.sample("x", lambda m, s: Normal(m, s), parents=[mu, sigma])
bn.compile()
# BN compiled: 3 nodes, 2 arcs, 528 CPT entries (0.004 MB)
```

BN structure — `mu` and `sigma` drive the conditional distribution of `x`:

![Continuous BN structure](img/continuous_bn.svg)

### Count variables (Poisson / Binomial)

```python
from distributions import Poisson, Uniform

bn   = BNppl()
rate = bn.sample("rate", Uniform(1.0, 5.0))
k    = bn.sample("k", lambda r: Poisson(r), parents=[rate])
bn.compile()
bn.query("rate", evidence={"k": 6})
# posterior of rate shifts toward higher values
```

Posterior P(rate | k=6) — observing a high count updates the rate belief upward:

![Poisson conditional inference](img/poisson_inference.svg)

---

## Notebooks

| Notebook | Content |
|---|---|
| [`Bnpyro_Tutorial.ipynb`](Bnpyro_Tutorial.ipynb) | 8 worked examples (discrete, continuous, plates, recursion, HOF) |
| [`Discretization.ipynb`](Discretization.ipynb) | Deep dive: n_bins, MIDPOINT vs INTEGRATION, CPT explosion, BIN_ADAPTIVE, memory limits |
| [`Pyro_vs_Bnpyro.ipynb`](Pyro_vs_Bnpyro.ipynb) | Side-by-side comparison with Pyro: discrete BN, plates/thunks, continuous MCMC vs exact BN |
| [`recurse_examples.ipynb`](recurse_examples.ipynb) | Examples using `bn.recurse` for dynamic Bayesian networks |

### Tutorial examples (`Bnpyro_Tutorial.ipynb`)

1. **Classic BN** — Rain → Wet, exact posterior
2. **Thunk (`!t` / `der`)** — Shared biased coin, belief update
3. **Discretization strategies** — `BIN_UNIFORM` vs `BIN_ADAPTIVE`, CPT size comparison
4. **Template BN (plate)** — N students sharing the same structure
5. **Universal lambda** — Continuous → Bernoulli, Continuous → Continuous, Categorical → Continuous
6. **Multi-parent discrete** — WetGrass CPT with nested `bn.where`
7. **Recursion (`fix`)** — Bernoulli Markov chain + Gaussian random walk
8. **Higher-order functions** — Parametric thunk, `apply_n`, reusable sensor constructor

---

## API Reference

### `BNppl(...)`

```python
BNppl(
    n_bins=10,                        # bins for continuous variables
    discretization_method=MIDPOINT,   # MIDPOINT | INTEGRATION
    bin_strategy=BIN_UNIFORM,         # BIN_UNIFORM | BIN_ADAPTIVE
    memory_warn_mb=50.0,              # warn if CPT total exceeds threshold
    memory_limit_mb=None,             # abort compile if exceeded
)
```

| Method | Returns | Description |
|---|---|---|
| `bn.sample(name, dist_or_cpt, parents=None, n_bins=None)` | `BNNode` | Add a random variable node |
| `bn.where(condition, p_true, p_false)` | `_BernoulliCPT` | Conditional CPT (nestable for multi-parent) |
| `bn.plate(name, dist_or_fn, parents=None)` | `BNPlate` | Thunk: each call creates an independent node |
| `bn.recurse(name, step_fn, n_steps)` | `list[BNNode]` | Unroll a recursive program |
| `bn.compile()` | — | Build and check the BN |
| `bn.query(target, evidence)` | `dict` | Exact posterior: `{label: prob}` |
| `bn.show()` | — | Print BN summary |
| `bn.show_graph()` | — | Visualize BN in template notation |
| `bn.gum_bn` | `gum.BayesNet` | Access the underlying pyAgrum BN |

### Supported distributions (`distributions.py`)

| Distribution | Variable type | Description |
|---|---|---|
| `Bernoulli(p)` | LabelizedVariable | P(X=True) = p |
| `Categorical(probs)` | LabelizedVariable | Discrete over {0, 1, ..., K-1} |
| `Normal(loc, scale)` | DiscretizedVariable | Discretized Gaussian |
| `Beta(a, b)` | DiscretizedVariable | Discretized Beta |
| `Gamma(concentration, rate)` | DiscretizedVariable | Discretized Gamma |
| `Uniform(low, high)` | DiscretizedVariable | Discretized Uniform |
| `Exponential(rate)` | DiscretizedVariable | Discretized Exponential |
| `LogNormal(loc, scale)` | DiscretizedVariable | Discretized LogNormal |
| `Poisson(rate)` | **RangeVariable** | Integer-valued {0, 1, ..., ppf(0.9999, rate)} |
| `Binomial(n, p)` | **RangeVariable** | Integer-valued {0, 1, ..., n} |

---

## Discretization

Continuous variables are approximated by a discrete histogram over `n_bins` equal-width intervals.

### Bin strategies

| Strategy | Behaviour | When to use |
|---|---|---|
| `BIN_UNIFORM` | All nodes get `n_bins` | Default — full precision everywhere |
| `BIN_ADAPTIVE` | Fewer bins for nodes with many parents (k=2 → `n_bins//2`, k≥3 → `max(3, n_bins//4)`) | Avoid CPT explosion with many parents |

### Discretization methods (for conditional nodes)

| Method | How | When better |
|---|---|---|
| `MIDPOINT` | Evaluates CPT at parent bin center | Linear relationships |
| `INTEGRATION` | Integrates CPT over parent bin (uniform) | Non-linear relationships (Jensen's inequality) |

### CPT size

$$\text{CPT entries} = n\_bins^{k+1}$$

where k = number of continuous parents. Use `BIN_ADAPTIVE` or per-node override `bn.sample(..., n_bins=5)` to keep this manageable.

### Memory protection

```python
bn = BNppl(n_bins=20, memory_warn_mb=10.0, memory_limit_mb=100.0)
bn.compile()
# [WARN] Total CPT memory: 12.50 MB > threshold 10.00 MB
# Top-3 largest nodes:  x (10.00 MB)  mu (1.25 MB)  sigma (1.25 MB)
# Tips: lower n_bins, use BIN_ADAPTIVE, or set memory_limit_mb
```

---

## Advanced patterns

### Nested `bn.where` — WetGrass

```python
bn        = BNppl()
rain      = bn.sample("rain",      Bernoulli(0.5))
sprinkler = bn.sample("sprinkler", bn.where(rain, 0.1, 0.5))
wetgrass  = bn.sample("wetgrass",  bn.where(rain,
    bn.where(sprinkler, 0.99, 0.90),
    bn.where(sprinkler, 0.90, 0.01),
))
bn.compile()
bn.query("rain", evidence={"wetgrass": True})
# {'False': 0.250, 'True': 0.750}   ← P(rain | wet) = 0.75 exactly
```

P(rain, sprinkler | wetgrass=True) — both causes updated simultaneously:

![WetGrass inference](img/wetgrass_inference.svg)

### Plate (thunk) — N i.i.d. coin flips

```python
theta = bn.sample("theta", Beta(2.0, 2.0))
coin  = bn.plate("flip", lambda t: Bernoulli(t), parents=[theta])
flips = [coin() for _ in range(4)]
bn.compile()

# Observe 3 heads, 1 tail → posterior of theta shifts up
evidence = {"flip_1": True, "flip_2": True, "flip_3": True, "flip_4": False}
bn.query("theta", evidence=evidence)
```

P(theta | 3 heads, 1 tail) — one shared parameter, N independent observations:

![Plate inference](img/plate_inference.svg)

### Temporal model (DBN via `recurse`)

```python
states = bn.recurse("X",
    lambda i, prev: Bernoulli(0.5) if prev is None
                    else bn.where(prev, 0.9, 0.1),
    n_steps=4
)
bn.compile()
bn.query("X_3", evidence={"X_0": True})
# {'False': 0.295, 'True': 0.705}   (started True, slowly diffusing toward 0.5)
```

P(X_1, X_2, X_3 | X_0=True) — belief propagates forward through the chain:

![DBN inference](img/dbn_inference.svg)

### Access all pyAgrum tools

```python
import pyagrum as gum
import pyagrum.lib.notebook as gnb

gnb.showBN(bn.gum_bn)              # visual graph in notebook
gnb.showInference(bn.gum_bn, evs={"wet": True}, targets=["rain"])
ie = gum.VariableElimination(bn.gum_bn)
```

---

## Project structure

```
Bnpyro.py                  # Main library
distributions.py           # Distribution classes (Normal, Beta, Poisson, ...)
Bnpyro_Tutorial.ipynb      # 8 worked examples
Discretization.ipynb       # Discretization deep dive
Pyro_vs_Bnpyro.ipynb       # Comparison with Pyro (discrete, plates, continuous)
recurse_examples.ipynb     # Recursion / DBN examples
img/                       # BN graph visualizations (generated by pyAgrum)
```

---

## References

- Faggian C., Pautasso D., Vanoni G. — *Higher-Order Bayesian Networks, Exactly*, POPL 2024
- Gonzales C., Wuillemin P.-H. — *pyAgrum*, 2020 — https://agrum.gitlab.io/
- Bingham et al. — *Pyro: Deep Universal Probabilistic Programming*, JMLR 2019
