"""
bnpyro.py — Compiles probabilistic programs into exact Bayesian Networks

A layer on top of pyAgrum that intercepts each probabilistic call,
builds the corresponding BN in parallel, and enables EXACT inference
via Variable Elimination (LazyPropagation). 
It's base on the article "Higher Order Bayesian Networks, Exactly" by 
Claudia FAGGIAN, Daniele PAUTASSO and Gabriele VANONI.

Usage:
    bn   = BNContext()
    rain = bn.sample("rain", dist.Bernoulli(0.2))
    wet  = bn.sample("wet",  bn.where(rain, 0.7, 0.01))
    coin = bn.thunk("coin", bn.where(rain, 0.8, 0.3))  # "!" from λ!-calculus
    y1   = coin()                                        # "der" #1
    y2   = coin()                                        # "der" #2

    p = bn.query("rain", evidence={"wet": True})

Correspondence with λ!-calculus (Faggian, Pautasso, Vanoni - POPL 2024):
    bn.sample("x", Bernoulli(p))           <->  let x = sample_d
    bn.sample("x", bn.where(parents, ...)) <->  let x = case⟨parents⟩
    bn.thunk("f", dist)                    <->  let f = !t
    f()                                    <->  der f
    for i in bn.plate("s", N): ...         <->  Template BN

Dependencies:
    pip install pyagrum pyro-ppl torch
    pip install scipy   # required only for discretization_method=INTEGRATION
"""

from __future__ import annotations

import torch
import pyro.distributions as dist
import pyagrum as gum
import numpy as np
from itertools import product as iproduct
from typing import Optional, Callable, Union


# CONSTANTS: discretization methods

MIDPOINT    = "midpoint"
INTEGRATION = "integration"


# Utilities

def _get_dist_range(distribution) -> tuple[float, float]:
    """Returns (lo, hi) covering ~99% of the probability mass of a continuous distribution."""
    def _f(x):
        return float(x.item() if hasattr(x, "item") else x)

    if isinstance(distribution, dist.Normal):
        mu, sigma = _f(distribution.loc), _f(distribution.scale)
        return mu - 3 * sigma, mu + 3 * sigma

    if isinstance(distribution, dist.Beta):
        a, b = _f(distribution.concentration1), _f(distribution.concentration0)
        mu    = a / (a + b)
        sigma = np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
        return max(0.001, mu - 3 * sigma), min(0.999, mu + 3 * sigma)

    if isinstance(distribution, dist.Gamma):
        a, r  = _f(distribution.concentration), _f(distribution.rate)
        mu    = a / r
        sigma = np.sqrt(a) / r
        return max(0.001, mu - 3 * sigma), mu + 3 * sigma

    if isinstance(distribution, dist.Uniform):
        return _f(distribution.low), _f(distribution.high)

    samples = distribution.sample((10_000,)).numpy()
    return float(np.percentile(samples, 1)), float(np.percentile(samples, 99))

def _dist_to_cpt(distribution, ticks: np.ndarray) -> list[float]:
    """Computes P(X in bin_i) for each bin via CDF (or Monte Carlo fallback), normalized."""
    probs = []
    try:
        for i in range(len(ticks) - 1):
            lo, hi = float(ticks[i]), float(ticks[i + 1])
            p = float(distribution.cdf(torch.tensor(hi, dtype=torch.float32)).item()) \
              - float(distribution.cdf(torch.tensor(lo, dtype=torch.float32)).item())
            probs.append(max(0.0, p))
    except (NotImplementedError, AttributeError):
        samples = distribution.sample((10_000,)).numpy()
        counts, _ = np.histogram(samples, bins=ticks)
        probs = counts.tolist()
    total = sum(probs)
    return [p / total for p in probs] if total > 0 else [1.0 / len(probs)] * len(probs)

def _discretize_continuous(name: str, distribution, n_bins: int = 10):
    """Creates a pyAgrum DiscretizedVariable for a continuous distribution without parents."""
    lo, hi = _get_dist_range(distribution)
    ticks = np.linspace(lo, hi, n_bins + 1)
    var = gum.DiscretizedVariable(name, name)
    for tick in ticks:
        var.addTick(float(tick))
    return var, ticks

# Nodes

class BNNode:
    """A random variable node in a BNContext."""

    def __init__(self, name: str, is_continuous: bool = False,
                 ticks: Optional[np.ndarray] = None):
        self.name          = name
        self.is_continuous = is_continuous
        self.ticks         = ticks

    def __repr__(self):
        kind = "continuous" if self.is_continuous else "discrete"
        return f"BNNode({self.name!r}, {kind})"

    def __bool__(self):
        raise TypeError(
            f"BNNode '{self.name}' cannot be converted to Python bool.\n"
            f"Use bn.where(condition, p_true, p_false) to build conditional distributions."
        )

class BNThunk:
    """
    Thunk from λ!-calculus: a frozen, reusable distribution.
    each call () instantiates a new node (der).

    Two modes:
      Simple     - bn.thunk("f", dist.Bernoulli(0.5))                     zero-arg thunk
      Parametric - bn.thunk("f", lambda b: dist.Bernoulli(b), [bias_node]) thunk of one argument
    """

    def __init__(self, ctx: "BNContext", base_name: str,
                 dist_or_fn, fn_parents: Optional[list] = None):
        self._ctx        = ctx
        self._base_name  = base_name
        self._dist_or_fn = dist_or_fn
        self._fn_parents = fn_parents or []
        self._call_count = 0

    def __call__(self, name: Optional[str] = None) -> BNNode:
        self._call_count += 1
        node_name = name or f"{self._base_name}_{self._call_count}"
        full_name = self._ctx._full_name(node_name)
        d = self._dist_or_fn

        if self._fn_parents and callable(d) and not isinstance(d, _BernoulliCPT):
            # Parametric thunk: lambda + explicit parents
            node = self._ctx._add_from_fn(full_name, d, self._fn_parents)
        elif isinstance(d, _BernoulliCPT):
            node = self._ctx._add_bernoulli_cpt(full_name, d.cond)
        elif isinstance(d, dist.Bernoulli):
            p = float(d.probs.item() if hasattr(d.probs, "item") else d.probs)
            node = self._ctx._add_bernoulli_root(full_name, p)
        else:
            node = self._ctx._add_continuous(full_name, d)

        self._ctx._thunk_derived.add(full_name)
        self._ctx._thunk_groups.setdefault(self._base_name, []).append(node.name)
        return node

    def __repr__(self):
        return f"BNThunk({self._base_name!r}, calls={self._call_count})"

class BNPair:
    """
    Ordered pair of BNNodes: corresponds to ⊗ (tensor product) in λ!-calculus.

    Supports Python tuple destructuring:
        x, y = bn_pair

    Notes:
        pyAgrum does not support tuple-valued nodes, so BNPair is simply
        a structured container for two existing BNNodes. The joint
        distribution is represented implicitly by their shared parents.
    """

    def __init__(self, first: BNNode, second: BNNode):
        self.first  = first
        self.second = second

    def __iter__(self):
        yield self.first
        yield self.second

    def __getitem__(self, i: int) -> BNNode:
        return (self.first, self.second)[i]

    def __repr__(self):
        return f"({self.first.name}, {self.second.name})"


# Internal structures

class _BernoulliCPT:
    def __init__(self, cond: "_Conditional"):
        self.cond = cond

class _Conditional:
    def __init__(self, parents: list[BNNode], p_true, p_false):
        self.parents = parents
        self.p_true  = p_true
        self.p_false = p_false

def _collect_parents(cond: "_Conditional") -> list[BNNode]:
    """
    Collects all unique parent BNNodes from a potentially nested CPT.
    Preserves order of appearance (first parent found = index 0 in CPT).
    """
    seen: set[str] = set()
    result: list[BNNode] = []

    def _visit(c: "_Conditional"):
        for p in c.parents:
            if p.name not in seen:
                seen.add(p.name)
                result.append(p)
        for branch in (c.p_true, c.p_false):
            if isinstance(branch, _BernoulliCPT):
                _visit(branch.cond)

    _visit(cond)
    return result

def _eval_cond(cond: "_Conditional", assignment: dict) -> float:
    """
    Evaluates P(X=True | assignment) by traversing the tree of nested bn.where calls.
    assignment: {parent_name: 0 (False) | 1 (True)}
    """
    parent = cond.parents[0]
    branch = cond.p_true if assignment[parent.name] == 1 else cond.p_false
    if isinstance(branch, _BernoulliCPT):
        return _eval_cond(branch.cond, assignment)
    return float(branch)


# Plate iterator

class _PlateIterator:
    """Iterator for bn.plate(): prefixes node names at each iteration."""

    def __init__(self, ctx: "BNContext", name: str, size: int):
        self._ctx  = ctx
        self._name = name
        self._size = size

    def __iter__(self):
        for i in range(self._size):
            self._ctx._plate_prefix.append(f"{self._name}_{i}")
            yield i
            self._ctx._plate_prefix.pop()


# MAIN CONTEXT

class BNContext:
    """
    Context for compiling a probabilistic program into a pyAgrum Bayesian Network.

    Parameters:
        n_bins                 : number of bins for discretization (default: 10)
        discretization_method  : MIDPOINT (default) or INTEGRATION

    Discretization method for continuous nodes with continuous parents:
        MIDPOINT    — evaluates the distribution at each parent bin center.
                      Fast; accurate for fine bins.
        INTEGRATION — numerically integrates P(X in bin_i | Y in bin_j) via scipy.
                      More accurate for coarse bins; requires scipy.

    The method can be changed at any time:
        bn.discretization_method = INTEGRATION

    Examples:
        bn   = BNContext()
        rain = bn.sample("rain", dist.Bernoulli(0.2))
        wet  = bn.sample("wet",  bn.where(rain, 0.9, 0.01))

        # Continuous node without parents
        temp = bn.sample("temp", dist.Normal(20.0, 5.0))

        # Continuous node conditioned by continuous parent
        mu = bn.sample("mu", dist.Normal(0.0, 1.0))
        x  = bn.sample("x", lambda mu_val: dist.Normal(mu_val, 1.0), parents=[mu])

        p = bn.query("rain", evidence={"wet": True})
    """

    def __init__(self, n_bins: int = 10, discretization_method: str = MIDPOINT):
        if discretization_method not in (MIDPOINT, INTEGRATION):
            raise ValueError(
                f"discretization_method must be '{MIDPOINT}' or '{INTEGRATION}'"
            )
        self._gum_bn               = gum.BayesNet("HigherOrderBN")
        self._nodes: dict[str, BNNode] = {}
        self._n_bins               = n_bins
        self._plate_prefix: list[str] = []
        self._discretization_method = discretization_method
        self._plates: dict[str, int] = {}             # {plate_name: size}
        self._thunk_derived: set[str] = set()         # node names created by thunk()()
        self._thunk_groups: dict[str, list[str]] = {} # {base_name: [node_name, ...]}

    # To change de discretization method after construction
    @property
    def discretization_method(self) -> str:
        return self._discretization_method

    @discretization_method.setter
    def discretization_method(self, value: str):
        if value not in (MIDPOINT, INTEGRATION):
            raise ValueError(
                f"discretization_method must be '{MIDPOINT}' or '{INTEGRATION}'"
            )
        self._discretization_method = value

    # sample
    def sample(self, name: str,
               distribution_or_fn: Union[object, Callable],
               parents: Optional[list[BNNode]] = None) -> BNNode:
        """
        Creates a node in the BN.

        Supported cases:
            bn.sample("rain", dist.Bernoulli(0.2))
            bn.sample("wet",  bn.where(rain, 0.9, 0.01))
            bn.sample("temp", dist.Normal(20.0, 5.0))
            bn.sample("x", lambda mu: dist.Normal(mu, 1.0), parents=[mu_node])
            bn.sample("x", lambda mu, nu: dist.Normal(mu, nu),parents=[mu_node, nu_node])
        """
        full_name = self._full_name(name)

        # Callable + parents -> probe return type and route
        if callable(distribution_or_fn) and parents is not None:
            return self._add_from_fn(full_name, distribution_or_fn, parents)

        d = distribution_or_fn

        if isinstance(d, _BernoulliCPT):
            return self._add_bernoulli_cpt(full_name, d.cond)

        if isinstance(d, dist.Bernoulli):
            p = float(d.probs.item() if hasattr(d.probs, "item") else d.probs)
            return self._add_bernoulli_root(full_name, p)

        if isinstance(d, dist.Categorical):
            return self._add_categorical_root(full_name, d.probs.tolist())

        # All other distributions treated as continuous (with fallback discretization)
        return self._add_continuous(full_name, d)

    # thunk
    def thunk(self, name: str, distribution_or_fn,
              parents: Optional[list] = None) -> BNThunk:
        """
        Freezes a distribution - corresponds to !t in λ!-calculus.
        Each call () instantiates a new node (der).

        Simple (zero-arg thunk):
            coin = bn.thunk("coin", bn.where(bias, 0.8, 0.3))
            coin = bn.thunk("coin", dist.Bernoulli(0.5))

        Parametric (thunk of one argument — higher-order):
            coin = bn.thunk("coin", lambda b: dist.Bernoulli(b), parents=[bias_node])
            Each coin() creates a node ~ Bernoulli(bias_node) via the lambda.
        """
        return BNThunk(self, name, distribution_or_fn, parents)

    # plate
    def plate(self, name: str, size: int) -> _PlateIterator:
        """
        Repeats a BN structure N times (Template BN, corresponds to plate notation).

        Usage:
            for i in bn.plate("students", 5):
                skill  = bn.sample("skill",  dist.Bernoulli(0.6))
                result = bn.sample("result", bn.where(skill, 0.9, 0.1))
        """
        self._plates[name] = size
        return _PlateIterator(self, name, size)

    # recurse
    def recurse(self, name: str, step_fn: Callable, n_steps: int) -> list:
        """
        Encodes a recursive probabilistic program as a chain BN.
        Corresponds to (fix f) applied n_steps times.

        step_fn(i: int, prev: Optional[BNNode]) -> distribution | _BernoulliCPT | Callable
            i=0, prev=None  : base case  - returns an unconditional distribution
            i>0, prev=BNNode: step case  - returns a distribution or CPT depending on prev
                              if a Callable is returned, it is called with parents=[prev]

        Returns a list of BNNodes [node_0, ..., node_{n_steps-1}].

        Example - Bernoulli Markov chain:
            states = bn.recurse("X",
                lambda _, prev: dist.Bernoulli(0.5) if prev is None
                                else bn.where(prev, 0.9, 0.1),
                n_steps=4
            )

        Example - Gaussian random walk:
            pos = bn.recurse("pos",
                lambda _, prev: dist.Normal(0.0, 1.0) if prev is None
                                else (lambda p: dist.Normal(p, 0.1)),
                n_steps=5
            )
        """
        nodes: list[BNNode] = []
        for i in range(n_steps):
            prev      = nodes[-1] if nodes else None
            node_name = f"{name}_{i}"
            d_or_fn   = step_fn(i, prev)
            if callable(d_or_fn) and not isinstance(d_or_fn, _BernoulliCPT) and prev is not None:
                node = self.sample(node_name, d_or_fn, parents=[prev])
            else:
                node = self.sample(node_name, d_or_fn)
            nodes.append(node)
        return nodes

    # where
    def where(self, condition: BNNode, p_true, p_false) -> _BernoulliCPT:
        """
        Builds a conditional Bernoulli CPT (replaces torch.where).

        p_true and p_false can be floats OR nested bn.where calls,
        allowing multi-parent CPTs:

            # 1 parent
            wet = bn.sample("wet", bn.where(rain, 0.9, 0.01))

            # 2 parents (nested)
            wet = bn.sample("wet", bn.where(rain,
                bn.where(sprinkler, 0.99, 0.90),   # rain=True
                bn.where(sprinkler, 0.10, 0.01)    # rain=False
            ))
        """
        if not isinstance(condition, BNNode):
            raise TypeError(f"condition must be a BNNode, got {type(condition)}")
        return _BernoulliCPT(_Conditional([condition], p_true, p_false))

    # pair / letp
    def pair(self, node1: BNNode, node2: BNNode) -> BNPair:
        """
        Creates a BNPair - corresponds to ⊗-introduction in λ!-calculus.

            weather = bn.pair(rain, wind)   <->   (rain, wind) : B ⊗ B
        """
        return BNPair(node1, node2)

    def letp(self, pair_expr: BNPair, body: Callable) -> "BNNode | BNPair":
        """
        Eliminates a BNPair - corresponds to letp in λ!-calculus.

            letp (x, y) = e in f(x, y)

        Usage:
            wet = bn.letp(weather, lambda rain, wind:
                      bn.sample("wet", bn.where(rain, 0.9, 0.01)))

        Note: Python tuple destructuring (x, y = pair) is equivalent;
              bn.letp() makes the correspondence with the paper explicit.
        """
        x, y = pair_expr
        return body(x, y)

    # query
    def query(self, target: str, evidence: Optional[dict] = None) -> dict:
        """
        Exact inference via LazyPropagation.
        Returns the posterior distribution of target as dict label->prob.
        """
        ie = gum.LazyPropagation(self._gum_bn)

        if evidence:
            gum_ev = {}
            for var_name, val in evidence.items():
                full = var_name if var_name in self._gum_bn.names() \
                       else self._full_name(var_name)
                if isinstance(val, bool):
                    gum_ev[full] = "True" if val else "False"
                elif isinstance(val, (int, float)):
                    gum_ev[full] = int(val)
                else:
                    gum_ev[full] = val
            ie.setEvidence(gum_ev)

        ie.makeInference()
        target_full = target if target in self._gum_bn.names() \
                      else self._full_name(target)
        posterior = ie.posterior(target_full)
        var       = self._gum_bn.variable(target_full)
        return {var.label(i): float(posterior[{target_full: i}])
                for i in range(var.domainSize())}

    def show(self):
        print(f"\nBN compilé : {len(self._gum_bn.nodes())} noeuds, "
              f"{len(self._gum_bn.arcs())} arcs  "
              f"[méthode={self._discretization_method}]")
        print("Nœuds :", list(self._gum_bn.names()))
        print("Arcs  :", [(self._gum_bn.variable(a).name(),
                           self._gum_bn.variable(b).name())
                          for a, b in self._gum_bn.arcs()])

    # factor semantics 

    def _val_to_idx(self, node_name: str, val) -> int:
        """Converts a user-provided value to a pyAgrum variable index."""
        var  = self._gum_bn.variable(node_name)
        node = self._nodes.get(node_name)
        if isinstance(val, bool):
            return 1 if val else 0
        if isinstance(val, float) and node and node.is_continuous:
            ticks = node.ticks
            for i in range(len(ticks) - 1):
                if ticks[i] <= val < ticks[i + 1]:
                    return i
            return len(ticks) - 2   # last bin (val == ticks[-1])
        if isinstance(val, str):
            return var.index(val)
        return int(val)

    def prob(self, assignment: dict) -> float:
        """
        Joint probability P(X₁=x₁,...,Xₙ=xₙ) = ∏ᵢ P(Xᵢ=xᵢ | pa(Xᵢ)).

        This is the direct evaluation of the factor decomposition.
        assignment: {node_name: value}  (bool, int index, float, or label string)
        """
        p = 1.0
        for node_name in self._gum_bn.names():
            cpt      = self._gum_bn.cpt(node_name)
            inst     = gum.Instantiation(cpt)
            node_var = self._gum_bn.variable(node_name)
            inst.chgVal(node_var, self._val_to_idx(node_name, assignment[node_name]))
            node_id = self._gum_bn.idFromName(node_name)
            for pid in self._gum_bn.parents(node_id):
                pvar  = self._gum_bn.variable(pid)
                pname = pvar.name()
                if pname in assignment:
                    inst.chgVal(pvar, self._val_to_idx(pname, assignment[pname]))
            p_i = float(cpt[inst])
            if p_i <= 0.0:
                return 0.0
            p *= p_i
        return p

    def log_prob(self, assignment: dict) -> float:
        """
        Log joint probability log P(X=x) = Σᵢ log P(Xᵢ=xᵢ | pa(Xᵢ)).
        Returns -inf if any factor is zero.
        """
        import math
        lp = 0.0
        for node_name in self._gum_bn.names():
            cpt      = self._gum_bn.cpt(node_name)
            inst     = gum.Instantiation(cpt)
            node_var = self._gum_bn.variable(node_name)
            inst.chgVal(node_var, self._val_to_idx(node_name, assignment[node_name]))
            node_id = self._gum_bn.idFromName(node_name)
            for pid in self._gum_bn.parents(node_id):
                pvar  = self._gum_bn.variable(pid)
                pname = pvar.name()
                if pname in assignment:
                    inst.chgVal(pvar, self._val_to_idx(pname, assignment[pname]))
            p_i = float(cpt[inst])
            if p_i <= 0.0:
                return float('-inf')
            lp += math.log(p_i)
        return lp

    def evidence_prob(self, evidence: Optional[dict] = None) -> float:
        """
        Marginal probability P(evidence) = Σ_{others} ∏ᵢ P(Xᵢ=xᵢ | pa(Xᵢ)).
        Computed via Variable Elimination (pyAgrum LazyPropagation).

        Useful for model comparison (Bayes factors: P(e|M₁)/P(e|M₂)).
        Returns 1.0 if no evidence is provided.
        """
        ie = gum.LazyPropagation(self._gum_bn)
        if evidence:
            gum_ev: dict = {}
            for var_name, val in evidence.items():
                full = var_name if var_name in self._gum_bn.names() \
                       else self._full_name(var_name)
                if isinstance(val, bool):
                    gum_ev[full] = "True" if val else "False"
                elif isinstance(val, (int, float)):
                    gum_ev[full] = int(val)
                else:
                    gum_ev[full] = val
            ie.setEvidence(gum_ev)
        ie.makeInference()
        return float(ie.evidenceProbability())

    @property
    def gum_bn(self) -> gum.BayesNet:
        return self._gum_bn

    # visualization: plate notation

    def _parse_node_name(self, full_name: str):
        """
        Parses a node name into plate segments and local name.
        "student_0/skill" -> ([("student", 0)], "skill")
        "s_0/m_1/grade"   -> ([("s", 0), ("m", 1)], "grade")
        """
        parts = full_name.split('/')
        segs  = []
        for part in parts[:-1]:
            for pname in sorted(self._plates, key=len, reverse=True):
                if part.startswith(pname + '_'):
                    tail = part[len(pname) + 1:]
                    try:
                        segs.append((pname, int(tail)))
                        break
                    except ValueError:
                        pass
        return segs, parts[-1]

    def _to_template(self, full_name: str) -> str:
        """Returns the template name (all indices -> 0) of a node."""
        segs, local = self._parse_node_name(full_name)
        if not segs:
            return full_name
        return '/'.join(f"{p}_0" for p, _ in segs) + '/' + local

    def _cpt_summary(self, template_name: str) -> str:
        """Compact CPT summary for display (root nodes only)."""
        node = self._nodes.get(template_name)
        if node is None:
            return ""
        if node.is_continuous:
            return "cont."
        node_id = self._gum_bn.idFromName(template_name)
        if self._gum_bn.parents(node_id):
            return ""   # too complex to summarize
        try:
            var  = self._gum_bn.variable(template_name)
            cpt  = self._gum_bn.cpt(template_name)
            inst = gum.Instantiation(cpt)
            inst.setFirst()
            vals = []
            while not inst.end():
                vals.append(float(cpt[inst]))
                inst.inc()
            if var.domainSize() == 2:
                return f"p={vals[1]:.2f}"
            return "[" + ",".join(f"{v:.2f}" for v in vals) + "]"
        except Exception:
            return ""

    def show_graph(self, show_cpt: bool = False, figsize: tuple = (14, 8)) -> None:
        """
        Displays the BN in plate notation (template view) using matplotlib.

        show_cpt : display P(X=True) under Bernoulli root nodes.
        figsize  : matplotlib figure size.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyBboxPatch

        # 1. Template graph
        tmpl: dict[str, dict] = {}
        for full in self._gum_bn.names():
            segs, local = self._parse_node_name(full)
            if all(idx == 0 for _, idx in segs):
                is_thunk = full in self._thunk_derived
                tmpl[full] = dict(
                    label   = f"!{local}" if is_thunk else local,
                    is_thunk= is_thunk,
                    plate   = segs[0][0] if segs else None,
                    local   = local,
                )

        # Virtual plates for thunk groups with multiple dereferences
        # e.g. coin_1, coin_2 → plate "coin ×2" showing only coin_1
        thunk_remap: dict[str, str] = {}   # non-representative -> representative
        for base_name, members in self._thunk_groups.items():
            in_tmpl = [m for m in members if m in tmpl]
            if len(in_tmpl) > 1:
                rep = in_tmpl[0]
                for m in in_tmpl[1:]:
                    thunk_remap[m] = rep
                    del tmpl[m]
                tmpl[rep]["plate"] = base_name
                tmpl[rep]["label"] = f"!{base_name}"

        # Arcs (after thunk remap so non-representative nodes are already removed)
        def _resolve(n: str) -> str:
            return thunk_remap.get(n, self._to_template(n))

        arcs: list[tuple] = []
        seen_arcs: set    = set()
        for a, b in self._gum_bn.arcs():
            s = _resolve(self._gum_bn.variable(a).name())
            d = _resolve(self._gum_bn.variable(b).name())
            if (s, d) not in seen_arcs and s in tmpl and d in tmpl:
                seen_arcs.add((s, d))
                arcs.append((s, d))

        # Plates from bn.plate()
        plates_info = {
            pn: {"size": sz,
                 "members": [n for n, v in tmpl.items() if v["plate"] == pn]}
            for pn, sz in self._plates.items()
            if any(v["plate"] == pn for v in tmpl.values())
        }
        # Add virtual plates from thunk groups
        for base_name, members in self._thunk_groups.items():
            if len(members) > 1:
                rep_list = [m for m in [members[0]] if m in tmpl]
                if rep_list:
                    plates_info[base_name] = {"size": len(members),
                                               "members": rep_list}

        # 2. Layout: topological levels
        children = {n: [] for n in tmpl}
        in_deg   = {n: 0  for n in tmpl}
        for s, d in arcs:
            children[s].append(d)
            in_deg[d] += 1

        level: dict[str, int] = {}
        queue = [n for n in tmpl if in_deg[n] == 0]
        for n in queue:
            level[n] = 0
        visited = set(queue)
        while queue:
            nxt = []
            for n in queue:
                for c in children[n]:
                    level[c] = max(level.get(c, 0), level[n] + 1)
                    if c not in visited:
                        visited.add(c)
                        nxt.append(c)
            queue = nxt
        for n in tmpl:
            if n not in level:
                level[n] = 0

        by_level: dict[int, list] = {}
        for n, lv in level.items():
            by_level.setdefault(lv, []).append(n)

        X_STEP, Y_STEP = 3.5, 1.8
        pos: dict[str, tuple] = {}
        for lv, nodes in by_level.items():
            nodes_s = sorted(nodes, key=lambda n: (tmpl[n]["plate"] or "", n))
            y0 = (len(nodes_s) - 1) * Y_STEP / 2
            for i, n in enumerate(nodes_s):
                pos[n] = (lv * X_STEP, y0 - i * Y_STEP)

        # 3. Drawing
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('equal')
        ax.axis('off')

        RX, RY = 0.65, 0.38     # ellipse semi-axes
        PAD    = 0.55            # margin around plate members

        C_ROOT  = "#AED6F1"
        C_THUNK = "#FAD7A0"
        C_PLATE = "#EBF5FB"
        C_EDGE  = "#2874A6"

        # plate rectangles
        for pn, info in plates_info.items():
            mems = info["members"]
            if not mems:
                continue
            xs = [pos[n][0] for n in mems]
            ys = [pos[n][1] for n in mems]
            x0 = min(xs) - RX - PAD
            y0 = min(ys) - RY - PAD
            w  = max(xs) + RX + PAD - x0
            h  = max(ys) + RY + PAD - y0
            rect = FancyBboxPatch((x0, y0), w, h,
                                   boxstyle="round,pad=0.1",
                                   linewidth=1.8, edgecolor=C_EDGE,
                                   facecolor=C_PLATE, zorder=1)
            ax.add_patch(rect)
            ax.text(x0 + w - 0.1, y0 + h - 0.1,
                    f"{pn}  ×{info['size']}",
                    ha='right', va='top', fontsize=9,
                    color=C_EDGE, style='italic', zorder=4)

        # arcs
        def _ellipse_offset(rx, ry, vx, vy):
            L = (vx**2 + vy**2) ** 0.5
            if L < 1e-9:
                return 0.0
            ux, uy = vx / L, vy / L
            return 1.0 / ((ux / rx) ** 2 + (uy / ry) ** 2) ** 0.5

        for s, d in arcs:
            sx, sy = pos[s]
            dx, dy = pos[d]
            vx, vy = dx - sx, dy - sy
            L = (vx**2 + vy**2) ** 0.5
            if L < 1e-6:
                continue
            off_s = _ellipse_offset(RX, RY,  vx,  vy)
            off_d = _ellipse_offset(RX, RY, -vx, -vy)
            xs_  = sx + vx / L * off_s
            ys_  = sy + vy / L * off_s
            xd_  = dx - vx / L * off_d
            yd_  = dy - vy / L * off_d
            ax.annotate("", xy=(xd_, yd_), xytext=(xs_, ys_),
                        arrowprops=dict(arrowstyle="-|>", color="#2C3E50",
                                        lw=1.5, mutation_scale=16),
                        zorder=2)

        # nodes
        for n, data in tmpl.items():
            x, y = pos[n]
            color = C_THUNK if data["is_thunk"] else C_ROOT
            ell = mpatches.Ellipse((x, y), 2 * RX, 2 * RY,
                                    facecolor=color, edgecolor="#2C3E50",
                                    linewidth=1.5, zorder=3)
            ax.add_patch(ell)
            label = data["label"]
            if show_cpt:
                cpt_str = self._cpt_summary(n)
                ax.text(x, y + 0.06, label,
                        ha='center', va='center',
                        fontsize=9, fontweight='bold', zorder=5)
                if cpt_str:
                    ax.text(x, y - RY - 0.18, cpt_str,
                            ha='center', va='top',
                            fontsize=7, color='#555555', zorder=5)
            else:
                ax.text(x, y, label,
                        ha='center', va='center',
                        fontsize=9, fontweight='bold', zorder=5)

        # auto margins
        all_x = [p[0] for p in pos.values()]
        all_y = [p[1] for p in pos.values()]
        margin = 1.5
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

        fig.tight_layout()
        plt.show()

    # internal methods: base nodes
    def _full_name(self, name: str) -> str:
        if self._plate_prefix:
            return "/".join(self._plate_prefix) + "/" + name
        return name

    def _add_bernoulli_root(self, name: str, p: float) -> BNNode:
        var = gum.LabelizedVariable(name, name, 2)
        var.changeLabel(0, "False")
        var.changeLabel(1, "True")
        self._gum_bn.add(var)
        self._gum_bn.cpt(name).fillWith([1 - p, p])
        return self._register(name, False)

    def _add_categorical_root(self, name: str, probs: list) -> BNNode:
        var = gum.LabelizedVariable(name, name, len(probs))
        self._gum_bn.add(var)
        self._gum_bn.cpt(name).fillWith(probs)
        return self._register(name, False)

    def _add_bernoulli_cpt(self, name: str, cond: _Conditional) -> BNNode:
        var = gum.LabelizedVariable(name, name, 2)
        var.changeLabel(0, "False")
        var.changeLabel(1, "True")
        self._gum_bn.add(var)

        all_parents = _collect_parents(cond)
        for p in all_parents:
            self._gum_bn.addArc(p.name, name)

        # pyAgrum inserts each new parent at the head of the potential (last added = slowest).
        # Use Instantiation to read the actual order and fill without order assumptions.
        pot  = self._gum_bn.cpt(name)
        inst = gum.Instantiation(pot)
        inst.setFirst()
        while not inst.end():
            assignment = {p.name: inst.val(self._gum_bn.variable(p.name))
                          for p in all_parents}
            p_t   = _eval_cond(cond, assignment)
            x_val = inst.val(self._gum_bn.variable(name))   # 0=False, 1=True
            pot.set(inst, p_t if x_val == 1 else 1.0 - p_t)
            inst.inc()

        return self._register(name, False)

    def _add_continuous(self, name: str, distribution) -> BNNode:
        var, ticks = _discretize_continuous(name, distribution, self._n_bins)
        self._gum_bn.add(var)
        self._gum_bn.cpt(name).fillWith(_dist_to_cpt(distribution, ticks))
        return self._register(name, True, ticks)

    def _register(self, name: str, is_continuous: bool,
                  ticks: Optional[np.ndarray] = None) -> BNNode:
        node = BNNode(name, is_continuous, ticks)
        self._nodes[name] = node
        return node

    # helpers for nodes with parents via lambda
    def _repr_val(self, inst: "gum.Instantiation", p: BNNode) -> float:
        """Representative value of a parent in an Instantiation (midpoint or index)."""
        idx = inst.val(self._gum_bn.variable(p.name))
        if p.is_continuous:
            return float((p.ticks[idx] + p.ticks[idx + 1]) / 2)
        return float(idx)

    def _add_from_fn(self, name: str, dist_fn: Callable,
                     parents: list[BNNode]) -> BNNode:
        """
        Probes dist_fn to detect the returned distribution type,
        then routes to the appropriate construction method.

            lambda p: dist.Bernoulli(p)        -> _add_bernoulli_from_fn
            lambda p: dist.Categorical(probs)  -> _add_categorical_from_fn
            lambda p: dist.Normal(p, 1.0)      -> _add_continuous_conditional
        """
        probe_vals = [
            float((p.ticks[0] + p.ticks[-1]) / 2) if p.is_continuous else 0.0
            for p in parents
        ]
        probe_dist = dist_fn(*probe_vals)

        if isinstance(probe_dist, dist.Bernoulli):
            return self._add_bernoulli_from_fn(name, dist_fn, parents)
        if isinstance(probe_dist, dist.Categorical):
            return self._add_categorical_from_fn(name, dist_fn, parents)
        return self._add_continuous_conditional(name, dist_fn, parents)

    def _add_bernoulli_from_fn(self, name: str, dist_fn: Callable,
                                parents: list[BNNode]) -> BNNode:
        """Bernoulli node whose probability is a function of parents (any types)."""
        var = gum.LabelizedVariable(name, name, 2)
        var.changeLabel(0, "False")
        var.changeLabel(1, "True")
        self._gum_bn.add(var)
        for p in parents:
            self._gum_bn.addArc(p.name, name)

        pot  = self._gum_bn.cpt(name)
        inst = gum.Instantiation(pot)
        inst.setFirst()
        while not inst.end():
            parent_vals = [self._repr_val(inst, p) for p in parents]
            d   = dist_fn(*parent_vals)
            p_t = float(d.probs.item() if hasattr(d.probs, "item") else d.probs)
            x_val = inst.val(self._gum_bn.variable(name))
            pot.set(inst, p_t if x_val == 1 else 1.0 - p_t)
            inst.inc()

        return self._register(name, False)

    def _add_categorical_from_fn(self, name: str, dist_fn: Callable,
                                  parents: list[BNNode]) -> BNNode:
        """Categorical node (k values) whose probabilities are a function of parents."""
        probe_vals = [
            float((p.ticks[0] + p.ticks[-1]) / 2) if p.is_continuous else 0.0
            for p in parents
        ]
        k = len(dist_fn(*probe_vals).probs)

        var = gum.LabelizedVariable(name, name, k)
        self._gum_bn.add(var)
        for p in parents:
            self._gum_bn.addArc(p.name, name)

        pot  = self._gum_bn.cpt(name)
        inst = gum.Instantiation(pot)
        inst.setFirst()
        while not inst.end():
            parent_vals = [self._repr_val(inst, p) for p in parents]
            probs   = dist_fn(*parent_vals).probs.tolist()
            cat_val = inst.val(self._gum_bn.variable(name))
            pot.set(inst, probs[cat_val])
            inst.inc()

        return self._register(name, False)

    # conditional continuous node
    def _add_continuous_conditional(self, name: str, dist_fn: Callable,
                                     parents: list[BNNode]) -> BNNode:
        """
        Continuous variable X whose parameters depend on parents.
        dist_fn(*parent_values) -> PyTorch distribution.

        CPT is computed according to self.discretization_method:
            MIDPOINT    — evaluates dist_fn at each parent bin center
            INTEGRATION — numerically integrates over each parent bin (scipy)
        """
        # 1. Determine X's range by probing dist_fn at parent extremes
        probe_points: list[list[float]] = []
        for p in parents:
            if p.is_continuous:
                t = p.ticks
                probe_points.append([float(t[0]),
                                      float((t[0] + t[-1]) / 2),
                                      float(t[-1])])
            else:
                var = self._gum_bn.variable(p.name)
                probe_points.append([float(i) for i in range(var.domainSize())])

        all_lo, all_hi = [], []
        for combo in iproduct(*probe_points):
            lo, hi = _get_dist_range(dist_fn(*combo))
            all_lo.append(lo)
            all_hi.append(hi)

        x_ticks = np.linspace(min(all_lo), max(all_hi), self._n_bins + 1)

        # 2. Créer la DiscretizedVariable aGrUM
        var = gum.DiscretizedVariable(name, name)
        for tick in x_ticks:
            var.addTick(float(tick))
        self._gum_bn.add(var)

        # 3. Parent arcs -> X
        for p in parents:
            self._gum_bn.addArc(p.name, name)

        # 4. CPT via Instantiation - independent of pyAgrum's internal ordering
        pot   = self._gum_bn.cpt(name)
        inst  = gum.Instantiation(pot)
        cache: dict = {}
        inst.setFirst()
        while not inst.end():
            # Bin indices for each parent (in original order of `parents`)
            parent_bin_idxs = tuple(inst.val(self._gum_bn.variable(p.name))
                                    for p in parents)
            x_bin_idx = inst.val(self._gum_bn.variable(name))

            if parent_bin_idxs not in cache:
                combo = []
                for p, i in zip(parents, parent_bin_idxs):
                    if p.is_continuous:
                        lo = float(p.ticks[i])
                        hi = float(p.ticks[i + 1])
                        combo.append((lo, hi, (lo + hi) / 2.0, True))
                    else:
                        combo.append((float(i), float(i), float(i), False))

                cache[parent_bin_idxs] = (
                    self._cpt_midpoint(dist_fn, combo, x_ticks)
                    if self._discretization_method == MIDPOINT
                    else self._cpt_integration(dist_fn, combo, x_ticks)
                )

            pot.set(inst, cache[parent_bin_idxs][x_bin_idx])
            inst.inc()

        return self._register(name, True, x_ticks)

    def _cpt_midpoint(self, dist_fn: Callable,
                       parent_combo, x_ticks: np.ndarray) -> list[float]:
        """
        Midpoint approximation: evaluates dist_fn at the center of each parent bin.
        Fast; accuracy increases with n_bins.
        """
        midpoints = [info[2] for info in parent_combo]
        return _dist_to_cpt(dist_fn(*midpoints), x_ticks)

    def _cpt_integration(self, dist_fn: Callable,
                          parent_combo, x_ticks: np.ndarray) -> list[float]:
        """
        Numerical integration via scipy.integrate.nquad.

        For each bin of X:
            P(X in [x_lo, x_hi] | Y in [y_lo, y_hi])
                = (1/|bin_Y|) * integral_{y_lo}^{y_hi} [CDF(x_hi|y) - CDF(x_lo|y)] dy

        More accurate than midpoint for coarse bins.
        Discrete parents are treated as in midpoint (fixed value).
        """
        import warnings
        from scipy.integrate import nquad, IntegrationWarning
        warnings.filterwarnings("ignore", category=IntegrationWarning)

        n_bins_x         = len(x_ticks) - 1
        cont_indices     = [i for i, info in enumerate(parent_combo) if info[3]]
        cont_indices_set = set(cont_indices)
        disc_vals        = {i: info[2] for i, info in enumerate(parent_combo)
                            if not info[3]}
        ranges           = [(parent_combo[i][0], parent_combo[i][1])
                            for i in cont_indices]

        # No continuous parents: same as midpoint
        if not cont_indices:
            return self._cpt_midpoint(dist_fn, parent_combo, x_ticks)

        bin_vol = 1.0
        for i in cont_indices:
            bin_vol *= max(parent_combo[i][1] - parent_combo[i][0], 1e-12)

        def make_integrand(x_lo: float, x_hi: float):
            def integrand(*y_vals):
                args = []
                cont_iter = iter(y_vals)
                for i in range(len(parent_combo)):
                    if i in cont_indices_set:
                        args.append(next(cont_iter))
                    else:
                        args.append(disc_vals[i])
                d = dist_fn(*args)
                cdf_hi = float(d.cdf(torch.tensor(x_hi, dtype=torch.float32)).item())
                cdf_lo = float(d.cdf(torch.tensor(x_lo, dtype=torch.float32)).item())
                return max(0.0, cdf_hi - cdf_lo)
            return integrand

        probs = []
        for x_idx in range(n_bins_x):
            val, _ = nquad(
                make_integrand(float(x_ticks[x_idx]), float(x_ticks[x_idx + 1])),
                ranges
            )
            probs.append(val / bin_vol)

        total = sum(probs)
        return [p / total for p in probs] if total > 0 \
               else [1.0 / n_bins_x] * n_bins_x


# EXEMPLES

if __name__ == "__main__":

    print("EXEMPLE 1 : BN classique Rain -> Wet")

    bn1  = BNContext()
    rain = bn1.sample("rain", dist.Bernoulli(0.2))
    wet  = bn1.sample("wet",  bn1.where(rain, 0.7, 0.01))
    bn1.show()
    p = bn1.query("rain", evidence={"wet": True})
    print(f"\nP(rain | wet=True) = {p}")
    print(f"  -> P(rain=True | wet=True) ≈ {p['True']:.3f}  (attendu ≈ 0.946)")

    print("\nEXEMPLE 2 : Pièce biaisée partagée — THUNK (!)")
    print("Correspond à : let coin = !(case⟨bias⟩) dans le λ!-calcul")

    bn2  = BNContext()
    bias = bn2.sample("bias", dist.Bernoulli(0.6))
    coin = bn2.thunk("coin", bn2.where(bias, 0.8, 0.3))
    y1   = coin()
    y2   = coin()
    bn2.show()
    p2 = bn2.query("bias", evidence={"coin_1": True, "coin_2": True})
    print(f"\nP(bias | y1=True, y2=True) = {p2}")
    print(f"  -> Posterior ≈ {p2['True']:.3f}  (deux piles -> biais élevé)")

    print("\nEXEMPLE 3 : Variable CONTINUE sans parents — discrétisation automatique")

    bn3   = BNContext(n_bins=8)
    temp  = bn3.sample("temp",  dist.Normal(torch.tensor(20.0), torch.tensor(5.0)))
    humid = bn3.sample("humid", dist.Normal(torch.tensor(60.0), torch.tensor(10.0)))
    bn3.show()
    p3 = bn3.query("temp")
    print("\nDistribution a priori de temp :")
    for label, prob in sorted(p3.items()):
        print(f"  {label:30s} | {'█' * int(prob * 60)} {prob:.3f}")

    print("\nEXEMPLE 4 : Template BN avec plate (N étudiants)")

    bn4  = BNContext()
    diff = bn4.sample("difficulty", dist.Bernoulli(0.4))
    for i in bn4.plate("student", 3):
        skill  = bn4.sample("skill",  dist.Bernoulli(0.6))
        result = bn4.sample("result", bn4.where(skill, 0.9, 0.1))
    bn4.show()
    p4 = bn4.query("difficulty", evidence={
        "student_0/result": True,
        "student_1/result": True,
        "student_2/result": False,
    })
    print(f"\nP(difficulty | 2 succès, 1 échec) = {p4}")

    print("\nEXEMPLE 5 : Nœud continu conditionné par un parent continu")
    print("X suit Normal(mu, 1.0)  avec  mu suit Normal(0, 1)")
    print("Comparaison midpoint vs intégration\n")

    for method in (MIDPOINT, INTEGRATION):
        bn5 = BNContext(n_bins=6, discretization_method=method)
        mu  = bn5.sample("mu", dist.Normal(0.0, 1.0))
        x   = bn5.sample("x",  lambda mu_val: dist.Normal(mu_val, 1.0), parents=[mu])
        bn5.show()
        p5 = bn5.query("x")
        print(f"Distribution marginale de X [{method}] :")
        for label, prob in sorted(p5.items()):
            print(f"  {label:35s} | {'█' * int(prob * 50)} {prob:.4f}")
        print()

    print("Changement de méthode à la volée :")
    bn6 = BNContext(n_bins=6)
    print(f"  méthode initiale  : {bn6.discretization_method}")
    bn6.discretization_method = INTEGRATION
    print(f"  méthode modifiée  : {bn6.discretization_method}")

    print("\nEXEMPLE 6 : bn.where avec plusieurs parents (rain + sprinkler -> wet)")
    print("CPT classique du réseau Wet Grass à 2 parents discrets\n")

    bn7       = BNContext()
    rain      = bn7.sample("rain",      dist.Bernoulli(0.2))
    sprinkler = bn7.sample("sprinkler", dist.Bernoulli(0.1))
    wet       = bn7.sample("wet", bn7.where(rain,
        bn7.where(sprinkler, 0.99, 0.90),   # rain=True
        bn7.where(sprinkler, 0.80, 0.01),   # rain=False
    ))
    bn7.show()

    p7 = bn7.query("rain", evidence={"wet": True})
    print(f"P(rain | wet=True)      = {p7['True']:.3f}")
    p7b = bn7.query("sprinkler", evidence={"wet": True})
    print(f"P(sprinkler | wet=True) = {p7b['True']:.3f}")

    print("\nEXEMPLE 7 : lambda universel — toutes combinaisons de types")

    print("\n  7a. Bernoulli conditionné par continu : coin ~ Bernoulli(biais)")
    bn8   = BNContext(n_bins=8)
    biais = bn8.sample("biais", dist.Uniform(torch.tensor(0.0), torch.tensor(1.0)))
    coin  = bn8.sample("coin",  lambda b: dist.Bernoulli(torch.tensor(b).clamp(0.01, 0.99)),
                        parents=[biais])
    bn8.show()
    p8 = bn8.query("biais", evidence={"coin": True})
    print("P(biais | coin=True) :", {k: f"{v:.3f}" for k, v in sorted(p8.items())})

    print("\n  7b. Continu conditionné par Catégoriel : val ~ Normal(cat*5, 1)")
    bn9 = BNContext(n_bins=6)
    cat = bn9.sample("cat", dist.Categorical(torch.tensor([0.3, 0.4, 0.3])))
    val = bn9.sample("val", lambda c: dist.Normal(torch.tensor(c * 5.0), torch.tensor(1.0)),
                     parents=[cat])
    bn9.show()
    p9 = bn9.query("val")
    print("Distribution de val :")
    for label, prob in sorted(p9.items()):
        print(f"  {label:20s} | {'█' * int(prob * 50)} {prob:.3f}")

    print("\nEXEMPLE 8 : Récursion — chaîne de Markov (Example 4.9 de l'article)")
    print("Correspond à : fix(λself.λn. if n=0 then sample(Bernoulli) else sample(case⟨self(n-1)⟩))\n")

    N = 4
    bn10   = BNContext()
    states = bn10.recurse("X",
        step_fn=lambda _, prev:
            dist.Bernoulli(0.5)        if prev is None   # base case  : X_0 ~ Ber(0.5)
            else bn10.where(prev, 0.9, 0.1),             # step case  : X_t | X_{t-1}
        n_steps=N
    )
    bn10.show()
    #bn10.show_graph()

    # P(X_3 = True | X_0 = True)
    p10 = bn10.query(f"X_{N-1}", evidence={"X_0": True})
    print(f"\nP(X_{N-1}=True | X_0=True)  = {p10['True']:.3f}  (attendu ~ 0.9^{N-1} = {0.9**(N-1):.3f})")

    print("\nEXEMPLE 8b : Récursion — random walk gaussien")
    bn11  = BNContext(n_bins=8)
    traj  = bn11.recurse("pos",
        step_fn=lambda _, prev:
            dist.Normal(0.0, 1.0)                    if prev is None
            else (lambda p: dist.Normal(p, 0.5)),    # X_t | X_{t-1} ~ N(X_{t-1}, 0.5)
        n_steps=4
    )
    bn11.show()
    #bn11.show_graph()

    print("\nEXEMPLE 9 : Fonctions d'ordre supérieur générales")
    print()

    print("  9a. Thunk paramétrique : coin ~ Bernoulli(biais) via lambda")
    print("      let coin = !(λb. Bernoulli(b))   —  thunk d'ordre supérieur\n")
    bn12  = BNContext(n_bins=8)
    biais = bn12.sample("biais", dist.Uniform(torch.tensor(0.0), torch.tensor(1.0)))
    # Parametric thunk: the frozen distribution depends on biais at call time
    coin  = bn12.thunk("coin", lambda b: dist.Bernoulli(torch.tensor(b).clamp(0.01, 0.99)),
                        parents=[biais])
    y1 = coin()   # coin_1 ~ Bernoulli(biais)
    y2 = coin()   # coin_2 ~ Bernoulli(biais) — nouvelle copie, même CPT
    bn12.show()
    #bn12.show_graph()
    p12 = bn12.query("biais", evidence={"coin_1": True, "coin_2": True})
    print(f"  P(biais > 0.5 | 2 piles) = {sum(v for k,v in p12.items() if float(k.split(';')[0].strip('[')) > 0.5):.3f}")

    print("\n  9b. HOF Python : fonction qui prend un BNThunk et l'applique N fois")
    print("      Correspond à  apply_n : !A → ℕ → A^n\n")

    def apply_n(thunk: BNThunk, n: int):
        """Higher-order: applies a thunk n times, returns all nodes."""
        return [thunk() for _ in range(n)]

    bn13   = BNContext()
    fair   = bn13.thunk("fair", dist.Bernoulli(0.5))
    flips  = apply_n(fair, 3)   # 3 independent fair coin flips
    bn13.show()
    #bn13.show_graph()

    print("\n  9c. HOF Python : constructeur de BN réutilisable (λx.t en pratique)")
    print("      make_sensor(signal) = bn.sample('sensor', λs.Normal(s,0.1), [signal])\n")

    def make_sensor(bn_ctx: BNContext, signal_node: BNNode, name: str) -> BNNode:
        """Reusable BN pattern: a noisy sensor reading a continuous signal."""
        return bn_ctx.sample(name, lambda s: dist.Normal(s, 0.2), parents=[signal_node])

    bn14   = BNContext(n_bins=6)
    signal = bn14.sample("signal", dist.Normal(0.0, 1.0))
    obs1   = make_sensor(bn14, signal, "obs1")
    obs2   = make_sensor(bn14, signal, "obs2")   # same pattern, different node
    bn14.show()
    #bn14.show_graph()
    # Evidence on continuous nodes uses bin index (integer), not string labels
    # bin 3 = positive half, bin 2 = negative half (out of 6 bins centered on 0)
    p14 = bn14.query("signal", evidence={"obs1": 3, "obs2": 3})
    print(f"  P(signal | obs1>0, obs2>0) : {[(k, f'{v:.3f}') for k, v in sorted(p14.items())]}")

    print("\nEXEMPLE 10 : Tuples et letp — type ⊗ du λ!-calcul")
    print("  Correspond à : letp (x, y) = e in f(x, y)\n")

    # 10a. bn.pair  = ⊗-introduction
    bn15  = BNContext()
    rain  = bn15.sample("rain",  dist.Bernoulli(0.3))
    wind  = bn15.sample("wind",  dist.Bernoulli(0.2))
    # weather : B ⊗ B  (pair of two Bernoulli nodes)
    weather = bn15.pair(rain, wind)
    print(f"  weather = {weather}")

    # 10b. bn.letp  = ⊗-elimination  (equivalent to Python: r, w = weather)
    wet = bn15.letp(weather, lambda r, w:
        bn15.sample("wet", bn15.where(r,
            bn15.where(w, 0.99, 0.90),   # rain=True
            bn15.where(w, 0.70, 0.01),   # rain=False
        ))
    )
    bn15.show()
    #bn15.show_graph(show_cpt=True)
    p15 = bn15.query("rain", evidence={"wet": True})
    print(f"  P(rain=True | wet=True) = {p15['True']:.3f}")

    # 10c. Python destructuring IS letp
    print("\n  Equivalence : bn.letp(...) ≡ Python destructuring")
    r2, w2 = weather                  # letp (r2, w2) = weather in ...
    print(f"  r2={r2.name}, w2={w2.name}  (même nœuds)")

    # 10d. Thunk returning a pair (higher-order + pair)
    print("\n  Thunk qui retourne une paire : !(λbias. (flip1, flip2))")
    bn16  = BNContext(n_bins=4)
    bias  = bn16.sample("bias", dist.Beta(torch.tensor(2.0), torch.tensor(2.0)))

    def make_pair_thunk(bias_node: BNNode):
        """Returns a thunk whose dereference creates a pair of correlated flips."""
        counter = [0]
        def _thunk():
            counter[0] += 1
            f1 = bn16.sample(f"flip{counter[0]}a",
                             lambda b: dist.Bernoulli(torch.tensor(b).clamp(0.01,0.99)),
                             parents=[bias_node])
            f2 = bn16.sample(f"flip{counter[0]}b",
                             lambda b: dist.Bernoulli(torch.tensor(b).clamp(0.01,0.99)),
                             parents=[bias_node])
            return bn16.pair(f1, f2)
        return _thunk

    pair_thunk = make_pair_thunk(bias)
    p1a, p1b  = pair_thunk()    # (flip1a, flip1b)
    p2a, p2b  = pair_thunk()    # (flip2a, flip2b)
    bn16.show()

    print("\nEXEMPLE 11 : Sémantique de facteurs")
    print("  P(X₁,...,Xₙ) = ∏ᵢ P(Xᵢ | pa(Xᵢ))  — chaque CPT est un facteur\n")

    # BN Rain -> Wet pour les démonstrations
    bn17  = BNContext()
    rain  = bn17.sample("rain", dist.Bernoulli(0.2))
    wet   = bn17.sample("wet",  bn17.where(rain, 0.9, 0.01))

    # 11a. Évaluation du joint (produit de facteurs)
    # P(rain=T, wet=T) = P(rain=T) * P(wet=T | rain=T) = 0.2 * 0.9 = 0.18
    p_joint = bn17.prob({"rain": True, "wet": True})
    print(f"  bn.prob(rain=T, wet=T)  = {p_joint:.4f}  (attendu 0.2×0.9 = {0.2*0.9:.4f})")

    p_joint2 = bn17.prob({"rain": False, "wet": False})
    print(f"  bn.prob(rain=F, wet=F)  = {p_joint2:.4f}  (attendu 0.8×0.99 = {0.8*0.99:.4f})")

    # Vérification : la somme de tous les joints = 1
    total = sum(
        bn17.prob({"rain": r, "wet": w})
        for r in (True, False)
        for w in (True, False)
    )
    print(f"  Σ tous joints           = {total:.6f}  (attendu 1.0)\n")

    # 11b. Log-probabilité
    lp = bn17.log_prob({"rain": True, "wet": True})
    print(f"  bn.log_prob(rain=T, wet=T) = {lp:.4f}  (= log({p_joint:.4f}) = {__import__('math').log(p_joint):.4f})\n")

    # 11c. Probabilité marginale de l'évidence  P(wet=True)
    p_wet   = bn17.evidence_prob({"wet": True})
    p_nowet = bn17.evidence_prob({"wet": False})
    print(f"  bn.evidence_prob(wet=T)  = {p_wet:.4f}  (attendu 0.2×0.9+0.8×0.01 = {0.2*0.9+0.8*0.01:.4f})")
    print(f"  bn.evidence_prob(wet=F)  = {p_nowet:.4f}")
    print(f"  Somme                    = {p_wet + p_nowet:.6f}  (attendu 1.0)\n")

    # 11d. Facteur de Bayes (comparaison de modèles)
    # Modèle M1 : P(rain) = 0.2   Modèle M2 : P(rain) = 0.5
    bn_m1 = BNContext()
    rain1 = bn_m1.sample("rain", dist.Bernoulli(0.2))
    wet1  = bn_m1.sample("wet",  bn_m1.where(rain1, 0.9, 0.01))

    bn_m2 = BNContext()
    rain2 = bn_m2.sample("rain", dist.Bernoulli(0.5))
    wet2  = bn_m2.sample("wet",  bn_m2.where(rain2, 0.9, 0.01))

    pe_m1 = bn_m1.evidence_prob({"wet": True})
    pe_m2 = bn_m2.evidence_prob({"wet": True})
    print(f"  Facteur de Bayes K = P(wet=T|M1)/P(wet=T|M2) = {pe_m1:.4f}/{pe_m2:.4f} = {pe_m1/pe_m2:.3f}")
    print(f"  → M2 (P(rain)=0.5) est {pe_m2/pe_m1:.1f}× plus probable sachant wet=True")
