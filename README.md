# Bnpyro

**Bnpyro** is a Python library that compiles higher-order probabilistic programs into exact Bayesian Networks (BNs). It uses [pyAgrum](https://agrum.gitlab.io/) for BN structure and inference, and is based on the formal framework of the **λ!-calculus** from:  

> Faggian, Pautasso & Vanoni — *Higher-Order Bayesian Networks, Exactly* (POPL 2024)  
> DOI: [10.1145/3632919](https://doi.org/10.1145/3632919)  

Instead of approximate Monte Carlo inference (as in Pyro/Stan), Bnpyro compiles the program into a BN and performs **exact inference** via Variable Elimination.  



## Features

| Bnpyro construct | λ!-calculus | Description |
|---|---|---|
| `bn.sample("x", dist)` | `let x = sample_d` | Declare a random variable |
| `bn.sample("x", bn.where(pa, ...))` | `case⟨pa⟩` | Conditional CPT |
| `bn.sample("x", lambda p: dist, parents=[p])` | extension | Universal lambda (any parent type) |
| `bn.thunk("f", dist)` / `f()` | `!t` / `der` | Freeze / thaw a distribution |
| `bn.plate("name", N)` | Template BN | Repeat a structure N times |
| `bn.recurse("x", fn, N)` | `fix f N` | Bounded recursion / Markov chains |
| Python HOF over `BNNode`/`BNThunk` | `λx.t` | Higher-order functions |
| `bn.pair(x, y)` / `bn.letp(p, fn)` | `x ⊗ y` / `letp` | Tensor product pairs |
| `bn.prob(assignment)` | factor semantics | Joint probability |
| `bn.query(target, evidence)` | VE | Exact posterior inference |

**Continuous variables** are automatically discretized into bins using the CDF (or Monte Carlo fallback). Two discretization methods are available: `MIDPOINT` (fast) and `INTEGRATION` (precise, requires `scipy`).  



## Installation

```bash
pip install pyagrum pyro-ppl torch matplotlib
pip install scipy   # only for INTEGRATION discretization
```

Clone or copy `src/Bnpyro.py` into your project.  


## Examples

The notebook `src/Bnpyro_Tutorial.ipynb` contains 10 worked examples:

1. **Classic BN** — Rain → Wet, exact posterior inference
2. **Thunk (!t / der)** — Shared biased coin, belief update
3. **Continuous variables** — Normal, Uniform, Gamma with automatic discretization
4. **Template BN (plate)** — N students sharing the same structure
5. **Universal lambda** — Continuous → Bernoulli, Continuous → Continuous, Categorical → Continuous
6. **Multi-parent discrete** — Wet Grass CPT with nested `bn.where`
7. **Recursion (fix)** — Bernoulli Markov chain + Gaussian random walk
8. **Higher-order functions** — Parametric thunk, `apply_n`, reusable sensor constructor
9. **Pairs and letp (⊗)** — Tensor product introduction and elimination
10. **Factor semantics** — `bn.prob`, `bn.log_prob`, `bn.evidence_prob`, Bayes Factor

## API Reference

### `BNContext(n_bins=10, discretization_method=MIDPOINT)`
Main compilation context. All methods are called on this object.

| Method | Returns | Description |
|---|---|---|
| `bn.sample(name, dist_or_cpt, parents=None)` | `BNNode` | Add a random variable node |
| `bn.where(condition, p_true, p_false)` | `_BernoulliCPT` | Conditional CPT (Bernoulli parent) |
| `bn.thunk(name, dist_or_fn, parents=None)` | `BNThunk` | Freeze a distribution |
| `bn.plate(name, size)` | iterator | Repeat structure N times |
| `bn.recurse(name, step_fn, n_steps)` | `list[BNNode]` | Unroll a recursive program |
| `bn.pair(node1, node2)` | `BNPair` | Create a tensor-product pair |
| `bn.letp(pair, fn)` | any | Destructure a pair |
| `bn.query(target, evidence)` | `dict` | Exact posterior via VE |
| `bn.prob(assignment)` | `float` | Joint probability |
| `bn.log_prob(assignment)` | `float` | Log joint probability |
| `bn.evidence_prob(evidence)` | `float` | Marginal probability of evidence |
| `bn.show()` | — | Print BN summary |
| `bn.show_graph(show_cpt=False)` | — | Visualize BN in template notation |
| `bn.gum_bn` | `gum.BayesNet` | Access the underlying pyAgrum BN |

### Supported distributions
Any Pyro/PyTorch distribution: `Bernoulli`, `Categorical`, `Normal`, `Beta`, `Gamma`, `Uniform`, `Exponential`, `LogNormal`, and more.

### Choosing `n_bins`
| Situation | Recommended `n_bins` |
|---|---|
| No continuous parents | 10 – 20 |
| 1 continuous parent | 10 – 15 |
| 2 continuous parents | 8 – 10 |
| 3+ continuous parents | 5 – 8 |


## Advanced patterns

### Multi-variable temporal model (DBN)
```python
prev = None
for t in range(4):
    if t == 0:
        loc   = bn.sample("loc_0",   dist.Normal(0.0, 1.0))
        speed = bn.sample("speed_0", dist.Bernoulli(0.5))
    else:
        loc_p, speed_p = prev
        loc   = bn.sample(f"loc_{t}",
                    lambda l, s: dist.Normal(l + 0.5*s, 0.3),
                    parents=[loc_p, speed_p])
        speed = bn.sample(f"speed_{t}", bn.where(speed_p, 0.8, 0.3))
    prev = (loc, speed)
```

### Higher-order reusable pattern
```python
def make_sensor(bn_ctx, signal, name):
    return bn_ctx.sample(name, lambda s: dist.Normal(s, 0.2), parents=[signal])

signal = bn.sample("signal", dist.Normal(0.0, 1.0))
obs1   = make_sensor(bn, signal, "obs1")
obs2   = make_sensor(bn, signal, "obs2")
```

### Access all pyAgrum tools
```python
import pyagrum as gum
import pyagrum.lib.notebook as gnb

# Any pyAgrum algorithm works on the compiled BN
ie = gum.VariableElimination(bn.gum_bn)
gnb.showBN(bn.gum_bn)   # pyAgrum's own visualizer (in notebooks)
```


## Project structure

```
src/
  Bnpyro.py              # Main library
  Bnpyro_Tutorial.ipynb  # Tutorial notebook (10 examples)
LaTeX/
  Bnpyro.tex             # Detailed documentation (French)
  main.tex               # Master LaTeX document
  Bibliographie.bib      # Bibliography
```


## References

- Faggian, C., Pautasso, D., Vanoni, G. — *Higher-Order Bayesian Networks, Exactly*, POPL 2024
- Gonzales, C., Wuillemin, P.-H. — *pyAgrum*, 2020 — https://agrum.gitlab.io/
- Bingham et al. — *Pyro: Deep Universal Probabilistic Programming*, JMLR 2019
