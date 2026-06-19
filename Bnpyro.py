"""
bnpyro.py — Compiles probabilistic programs into exact Bayesian Networks

A layer on top of pyAgrum that intercepts each probabilistic call,
builds the corresponding BN in parallel, and enables EXACT inference
via Variable Elimination (LazyPropagation). 
It's base on the article "Higher Order Bayesian Networks, Exactly" by 
Claudia FAGGIAN, Daniele PAUTASSO and Gabriele VANONI.

Usage:
    bn   = BNppl()
    rain = bn.sample("rain", dist.Bernoulli(0.2))
    wet  = bn.sample("wet",  bn.where(rain, 0.7, 0.01))
    coin = bn.plate("coin", bn.where(rain, 0.8, 0.3))  # "!" from λ!-calculus
    y1   = coin()                                       # "der" #1
    y2   = coin()                                       # "der" #2

    p = bn.query("rain", evidence={"wet": True})

Correspondence with λ!-calculus (Faggian, Pautasso, Vanoni - POPL 2024):
    bn.sample("x", Bernoulli(p))           <->  let x = sample_d
    bn.sample("x", bn.where(parents, ...)) <->  let x = case⟨parents⟩
    bn.plate("f", dist)                    <->  let f = !t
    f()                                    <->  der f

Dependencies:
    pip install pyagrum scipy matplotlib
"""

from __future__ import annotations

import logging
import warnings
import pyagrum as gum
import numpy as np
from dataclasses import dataclass, field
from itertools import product as iproduct
from typing import Optional, Callable, Union

logger = logging.getLogger(__name__)
from distributions import (
    Bernoulli, Categorical, Normal, Beta, Gamma,
    Uniform, Exponential, LogNormal, Poisson, Binomial,
)


# CONSTANTS: discretization methods

MIDPOINT    = "midpoint"
INTEGRATION = "integration"

# CONSTANTS: BNppl states

DESIGN   = "design"
COMPILED = "compiled"

# CONSTANTS: bin strategies

BIN_UNIFORM        = "uniform"         # all nodes get n_bins (default)
BIN_ADAPTIVE       = "adaptive"        # fewer bins for nodes with many continuous parents
BIN_MEMORY_BUDGET  = "memory_budget"   # each node targets an equal share of memory_warn_mb


# Utilities

def _get_dist_range(distribution) -> tuple[float, float]:
    """Returns (lo, hi) covering ~99% of the probability mass of a continuous distribution."""
    if isinstance(distribution, Normal):
        mu, sigma = distribution.loc, distribution.scale
        return mu - 3 * sigma, mu + 3 * sigma

    if isinstance(distribution, Beta):
        a, b = distribution.concentration1, distribution.concentration0
        mu = a / (a + b)
        sigma = np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
        return max(0.001, mu - 3 * sigma), min(0.999, mu + 3 * sigma)

    if isinstance(distribution, Gamma):
        a, r = distribution.concentration, distribution.rate
        mu = a / r
        sigma = np.sqrt(a) / r
        return max(0.001, mu - 3 * sigma), mu + 3 * sigma

    if isinstance(distribution, Uniform):
        return distribution.low, distribution.high

    samples = distribution.sample(10_000)
    return float(np.percentile(samples, 1)), float(np.percentile(samples, 99))

def _get_discrete_domain(distribution) -> tuple[int, int]:
    """Returns (min_k, max_k) for a discrete count distribution (Poisson / Binomial)."""
    if isinstance(distribution, Binomial):
        return 0, distribution.total_count
    if isinstance(distribution, Poisson):
        max_k = int(_poisson_ppf(0.9999, distribution.rate)) + 1
        return 0, max(max_k, 1)
    raise TypeError(f"Not a discrete count distribution: {type(distribution)}")

def _poisson_ppf(q: float, rate: float) -> int:
    from scipy.stats import poisson as _sp_poisson
    return int(_sp_poisson.ppf(q, rate))

def _dist_to_range_cpt(distribution, min_k: int, max_k: int) -> list[float]:
    """Computes normalized PMF for a discrete distribution over [min_k, max_k]."""
    probs = [max(0.0, distribution.pmf(k)) for k in range(min_k, max_k + 1)]
    total = sum(probs)
    return [p / total for p in probs] if total > 0 else [1.0 / len(probs)] * len(probs)

def _dist_to_cpt(distribution, ticks: np.ndarray) -> list[float]:
    """Computes P(X in bin_i) for each bin via CDF (or Monte Carlo fallback), normalized."""
    probs = []
    try:
        for i in range(len(ticks) - 1):
            lo, hi = float(ticks[i]), float(ticks[i + 1])
            p = distribution.cdf(hi) - distribution.cdf(lo)
            probs.append(max(0.0, p))
    except (NotImplementedError, AttributeError):
        samples = distribution.sample(10_000)
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
    """A random variable node in a BNppl."""

    def __init__(self, name: str, is_continuous: bool = False, ticks: Optional[np.ndarray] = None):
        self.name = name
        self.is_continuous = is_continuous
        self.ticks = ticks

    def __repr__(self):
        if self.is_continuous is None:
            kind = "pending"
        else:
            kind = "continuous" if self.is_continuous else "discrete"
        return f"BNNode({self.name!r}, {kind})"

    def __bool__(self):
        raise TypeError(
            f"BNNode '{self.name}' cannot be converted to Python bool.\n"
            f"Use bn.where(condition, p_true, p_false) to build conditional distributions."
        )

class BNPlate:
    """
    Plate from λ!-calculus: a frozen, reusable distribution (!t / bang).
    Each call () instantiates a new independent node (der).

    Two modes:
      Simple     - bn.plate("coin", dist.Bernoulli(0.5))
      Parametric - bn.plate("coin", lambda b: dist.Bernoulli(b), parents=[bias_node])
    """

    def __init__(self, ctx: "BNppl", base_name: str,
                 dist_or_fn, fn_parents: Optional[list] = None):
        self._ctx = ctx
        self._base_name = base_name
        self._dist_or_fn = dist_or_fn
        self._fn_parents = fn_parents or []
        self._call_count = 0

    def __call__(self, name: Optional[str] = None) -> BNNode:
        self._call_count += 1
        node_name = name or f"{self._base_name}_{self._call_count}"
        d = self._dist_or_fn

        if self._fn_parents and callable(d) and not isinstance(d, _BernoulliCPT):
            node = self._ctx.sample(node_name, d, parents=self._fn_parents)
        else:
            node = self._ctx.sample(node_name, d)

        self._ctx._plate_derived.add(node.name)
        self._ctx._plate_groups.setdefault(self._base_name, []).append(node.name)
        return node

    def __repr__(self):
        return f"BNPlate({self._base_name!r}, calls={self._call_count})"

# Internal structures

class _BernoulliCPT:
    def __init__(self, cond: "_Conditional"):
        self.cond = cond

class _Conditional:
    def __init__(self, parents: list[BNNode], p_true, p_false):
        self.parents = parents
        self.p_true = p_true
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


@dataclass
class _NodeSpec:
    """Pending node declaration stored during the DESIGN phase."""
    name: str # full node name (with plate prefix)
    dist_or_fn: object # distribution, _BernoulliCPT, or callable
    parents: list # list[BNNode] stubs (resolved at compile time)
    node: "BNNode" # stub returned to the user
    is_plate_derived: bool = False
    plate_base_name: str = None
    n_bins_override: Optional[int] = None  # per-node bin count (overrides strategy)
    labels: Optional[list] = None          # custom state labels (Bernoulli/Categorical)


# MAIN CONTEXT

class BNppl:
    """
    Context for compiling a probabilistic program into a pyAgrum Bayesian Network.

    Build phase:
        bn   = BNppl()
        rain = bn.sample("rain", Bernoulli(0.2))
        wet  = bn.sample("wet",  bn.where(rain, 0.9, 0.01))

    Compile phase (discretization options passed here):
        bn.compile(n_bins=10, estimate_proba_method=MIDPOINT)

    Query phase:
        p = bn.query("rain", evidence={"wet": True})
    """

    def __init__(self):
        # DESIGN state
        self._state: str = DESIGN
        self._specs: list[_NodeSpec] = []
        self._plate_derived: set[str] = set()
        self._plate_groups: dict[str, list] = {}
        # COMPILED state (populated by compile())
        self._gum_bn: Optional[gum.BayesNet] = None
        self._nodes:  dict[str, BNNode] = {}
        # Compilation options (set by compile())
        self._name: Optional[str] = None
        self._n_bins: Optional[int] = None
        self._bin_strategy: Optional[str] = None
        self._memory_warn_mb: Optional[float] = None
        self._memory_limit_mb: Optional[float] = None
        self._estimate_proba_method: Optional[str] = None
        self._budget_entries_per_node: Optional[float] = None

    # sample
    def sample(self, name: str, distribution_or_fn: Union[object, Callable], parents: Optional[list[BNNode]] = None, n_bins: Optional[int] = None, labels: Optional[list] = None) -> BNNode:
        """
        Declares a random variable node (DESIGN phase).
        The pyAgrum node is created lazily at compile() time.

        Raises RuntimeError if called in COMPILED state — call reopen() first.

        n_bins: optional per-node bin count override (continuous nodes only).
                Overrides both the global n_bins and the bin_strategy.

        Supported cases:
            bn.sample("rain", dist.Bernoulli(0.2))
            bn.sample("wet",  bn.where(rain, 0.9, 0.01))
            bn.sample("cat",  dist.Categorical(torch.tensor([0.3, 0.4, 0.3])))
            bn.sample("temp", dist.Normal(20.0, 5.0))
            bn.sample("temp", dist.Normal(20.0, 5.0), n_bins=20)
            bn.sample("x", lambda mu: dist.Normal(mu, 1.0), parents=[mu_node])
            bn.sample("x", lambda mu, nu: dist.Normal(mu, nu), parents=[mu_node, nu_node])
        """
        if self._state == COMPILED:
            raise RuntimeError(
                "Cannot declare nodes on a compiled BN. "
                "Call bn.reopen() first to return to DESIGN state."
            )
        stub = BNNode(name, is_continuous=None, ticks=None)
        self._specs.append(_NodeSpec(
            name=name,
            dist_or_fn=distribution_or_fn,
            parents=list(parents) if parents else [],
            node=stub,
            n_bins_override=n_bins,
            labels=list(labels) if labels else None,
        ))
        return stub

    # plate
    def plate(self, name: str, distribution_or_fn,
              parents: Optional[list] = None) -> BNPlate:
        """
        Freezes a distribution so it can be instantiated multiple times.
        Corresponds to !t (bang) in λ!-calculus; each call () is a deref (der).

        Simple (zero-arg):
            coin = bn.plate("coin", bn.where(bias, 0.8, 0.3))
            coin = bn.plate("coin", dist.Bernoulli(0.5))
            y1 = coin()   # creates coin_1
            y2 = coin()   # creates coin_2, independent of coin_1

        Parametric (one shared parent):
            coin = bn.plate("coin", lambda b: dist.Bernoulli(b), parents=[bias_node])
            Each coin() creates a node ~ Bernoulli(bias_node) via the lambda.
        """
        return BNPlate(self, name, distribution_or_fn, parents)

    # recurse
    def recurse(self, name_or_fn, step_fn_or_n, n_steps: int = None, *, order: int = 1, labels: Optional[list] = None):
        """
        Encodes a recursive probabilistic program as a chain BN.
        Corresponds to (fix f) applied n_steps times.

        Two modes:

        Single-node (backward-compatible):
            bn.recurse(name, step_fn, n_steps)
            step_fn(i, prev: Optional[BNNode]) -> dist | _BernoulliCPT | Callable
            Returns list[BNNode].

            Example - Bernoulli Markov chain:
                states = bn.recurse("X",
                    lambda _, prev: Bernoulli(0.5) if prev is None
                                    else bn.where(prev, 0.9, 0.1),
                    n_steps=4
                )

        Multi-node:
            bn.recurse(step_fn, n_steps, order=1)
            step_fn(t, prev) where prev = list of min(t, order) dicts,
                             prev[0] = most recent step {"name": BNNode, ...}
            Returns dict[str, list[BNNode]]: {"name": [node_0, ..., node_{n-1}], ...}

            Each element returned by step_fn: (local_name, dist_or_fn) or
                                              (local_name, dist_or_fn, parents)
            parents can mix BNNode (inter-step) and str (intra-step local name,
            must appear earlier in the same list).

            Example - position + velocity with intra-step arc:
                def step_fn(t, prev):
                    if not prev:
                        return [
                            ("vel", Bernoulli(0.5)),
                            ("loc", Normal(0.0, 1.0)),
                        ]
                    p = prev[0]
                    return [
                        ("vel", bn.where(p["vel"], 0.8, 0.3)),
                        ("loc", lambda l, v: Normal(l + v * 0.5, 0.1),
                                [p["loc"], p["vel"]]),
                    ]
                states = bn.recurse(step_fn, n_steps=4)
                # states["loc"] = [loc_0, loc_1, loc_2, loc_3]
                # states["vel"] = [vel_0, vel_1, vel_2, vel_3]
        """
        if isinstance(name_or_fn, str):
            return self._recurse_single(name_or_fn, step_fn_or_n, n_steps, labels)
        return self._recurse_multi(name_or_fn, step_fn_or_n, order, labels)

    def _recurse_single(self, name: str, step_fn: Callable, n_steps: int,
                        labels: Optional[list] = None) -> list:
        nodes: list[BNNode] = []
        for i in range(n_steps):
            prev = nodes[-1] if nodes else None
            d_or_fn = step_fn(i, prev)
            if callable(d_or_fn) and not isinstance(d_or_fn, _BernoulliCPT) and prev is not None:
                node = self.sample(f"{name}_{i}", d_or_fn, parents=[prev], labels=labels)
            else:
                node = self.sample(f"{name}_{i}", d_or_fn, labels=labels)
            nodes.append(node)
        return nodes

    def _recurse_multi(self, step_fn: Callable, n_steps: int, order: int,
                       labels: Optional[dict] = None) -> dict:
        history: list[dict] = []  # history[0] = most recent step {local_name: BNNode}
        result: dict[str, list] = {}

        for t in range(n_steps):
            prev = history[:order]
            step_specs = step_fn(t, prev)
            step_nodes: dict[str, BNNode] = {}

            for spec in step_specs:
                if len(spec) == 2:
                    local_name, dist_or_fn = spec
                    parents = []
                else:
                    local_name, dist_or_fn, parents = spec

                resolved = []
                for p in parents:
                    resolved.append(step_nodes[p] if isinstance(p, str) else p)

                node_labels = labels.get(local_name) if labels else None
                node = self.sample(f"{local_name}_{t}", dist_or_fn,
                                   parents=resolved if resolved else None,
                                   labels=node_labels)
                step_nodes[local_name] = node
                result.setdefault(local_name, []).append(node)

            history.insert(0, step_nodes)

        return result

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

    # query
    def query(self,
              target: Union[str, list],
              evidence: Optional[dict] = None,
              joint: bool = False) -> dict:
        """
        Exact inference via LazyPropagation.

        target  : a node name (str) or a list of node names.
        evidence: {node_name: value}  (bool, int index, or label string)
        joint   : if True and target is a list, returns the joint distribution
                  P(target[0], target[1], ... | evidence) as a dict with
                  tuple keys — e.g. {("True", "False"): 0.12, ...}.
                  If False (default), returns one marginal per target.

        Returns
        -------
        - Single target or joint=False  : {label: prob}  or  {node: {label: prob}}
        - joint=True with list target   : {(label_0, label_1, ...): prob}
        """
        self._ensure_compiled()
        ie = gum.LazyPropagation(self._gum_bn)

        if evidence:
            gum_ev = {}
            for var_name, val in evidence.items():
                if isinstance(val, bool):
                    gum_ev[var_name] = "True" if val else "False"
                elif isinstance(val, (int, float)):
                    gum_ev[var_name] = int(val)
                else:
                    gum_ev[var_name] = val
            ie.setEvidence(gum_ev)

        ie.makeInference()

        if joint and isinstance(target, list) and len(target) > 1:
            return self._joint_posterior_dict(ie, target)

        if isinstance(target, str):
            return self._posterior_dict(ie, target)
        return {t: self._posterior_dict(ie, t) for t in target}

    def _posterior_dict(self, ie, target: str) -> dict:
        posterior = ie.posterior(target)
        var = self._gum_bn.variable(target)
        return {var.label(i): float(posterior[{target: i}])
                for i in range(var.domainSize())}

    def _joint_posterior_dict(self, ie, targets: list) -> dict:
        ids = [self._gum_bn.idFromName(t) for t in targets]
        potential = ie.jointPosterior(set(ids))
        result = {}
        inst = gum.Instantiation(potential)
        inst.setFirst()
        while not inst.end():
            key = tuple(
                self._gum_bn.variable(ids[i]).label(inst.val(ids[i]))
                for i in range(len(targets))
            )
            result[key] = float(potential.get(inst))
            inst.inc()
        return result

    def __str__(self) -> str:
        self._ensure_compiled()
        arcs = [(self._gum_bn.variable(a).name(), self._gum_bn.variable(b).name())
                for a, b in self._gum_bn.arcs()]
        return (
            f"BN compiled: {len(self._gum_bn.nodes())} nodes, "
            f"{len(self._gum_bn.arcs())} arcs  "
            f"[method={self._estimate_proba_method}]\n"
            f"Nodes: {list(self._gum_bn.names())}\n"
            f"Arcs:  {arcs}"
        )

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def gum_bn(self) -> gum.BayesNet:
        self._ensure_compiled()
        return self._gum_bn

    # compilation 
    def _ensure_compiled(self) -> None:
        if self._state != COMPILED:
            raise RuntimeError(
                "BN is not compiled yet. Call bn.compile() before using "
                "query(), show(), show_graph(), or gum_bn."
            )

    def reopen(self) -> None:
        """
        Transitions COMPILED -> DESIGN.

        Discards the compiled pyAgrum BN (freeing memory) while keeping all
        existing node specs. A subsequent compile() or query() will rebuild
        the BN from scratch, which is useful after changing n_bins or
        discretization_method, or after adding new nodes.

        Example:
            bn.compile()                       # DESIGN -> COMPILED
            bn.reopen()                        # COMPILED -> DESIGN
            extra = bn.sample("extra", ...)    # add a node
            p = bn.query("extra")              # auto-recompiles
        """
        if self._state == DESIGN:
            return   # already in DESIGN, nothing to do
        self._gum_bn = None
        self._nodes = {}
        self._state = DESIGN

    def _resolve_n_bins(self, spec: _NodeSpec) -> int:
        """
        Returns the number of bins to use for a given node spec.

        Priority:
          1. Per-node override: bn.sample(..., n_bins=20)
          2. Strategy-specific formula (BIN_ADAPTIVE or BIN_MEMORY_BUDGET)
          3. Global self._n_bins (BIN_UNIFORM, default)

        BIN_ADAPTIVE:
            k continuous parents -> max(3, n_bins // 2^(k-1))  for k >= 2

        BIN_MEMORY_BUDGET:
            Targets equal memory per node. Budget = memory_warn_mb / n_nodes.
            For a node with k continuous parents and discrete-parent factor P:
                n_bins_i = floor( (budget_entries / P) ^ (1 / (k+1)) )
            Clamped to [3, n_bins].
        """
        if spec.n_bins_override is not None:
            return spec.n_bins_override

        if self._bin_strategy == BIN_ADAPTIVE:
            n_cont = sum(1 for p in spec.parents if p.is_continuous is True)
            if n_cont > 1:
                return max(3, self._n_bins // (2 ** (n_cont - 1)))

        if self._bin_strategy == BIN_MEMORY_BUDGET:
            n_cont = 0
            prod_disc = 1
            for p in spec.parents:
                    try:
                        prod_disc *= self._gum_bn.variable(p.name).domainSize()
                    except Exception:
                        #normalement self._specs est dans l'ordre topologique par contruction
                        if p.is_continuous : 
                            n_cont +=1
                        else:
                            prod_disc *= 2
            target = self._budget_entries_per_node / max(1, prod_disc)
            n = int(target ** (1.0 / (n_cont + 1)))
            return max(3, min(self._n_bins, n))

        return self._n_bins

    def compile(self,n_bins: int = 10,estimate_proba_method: str = MIDPOINT,
               bin_strategy: str = BIN_UNIFORM,memory_warn_mb: float = 50.0,
               memory_limit_mb: Optional[float] = None,name: str = "BN") -> None:
        """
        Transitions DESIGN -> COMPILED.

        Parameters
        ----------
        n_bins               : bins per continuous variable (default 10)
        estimate_proba_method: MIDPOINT (fast) or INTEGRATION (precise for coarse bins)
        bin_strategy         : BIN_UNIFORM (all nodes same), BIN_ADAPTIVE
                               (fewer bins for nodes with many continuous parents),
                               or BIN_MEMORY_BUDGET (equal memory share per node,
                               uses memory_warn_mb as the total budget)
        memory_warn_mb       : print a warning if total CPT size exceeds this (MB)
        memory_limit_mb      : raise RuntimeError if total CPT size exceeds this (MB)
        name                 : name of the underlying pyAgrum BayesNet
        """
        if self._state == COMPILED:
            return

        if estimate_proba_method not in (MIDPOINT, INTEGRATION):
            raise ValueError(
                f"estimate_proba_method must be '{MIDPOINT}' or '{INTEGRATION}'"
            )
        if bin_strategy not in (BIN_UNIFORM, BIN_ADAPTIVE, BIN_MEMORY_BUDGET):
            raise ValueError(
                f"bin_strategy must be '{BIN_UNIFORM}', '{BIN_ADAPTIVE}'"
                f" or '{BIN_MEMORY_BUDGET}'"
            )

        self._name = name
        self._n_bins = n_bins
        self._bin_strategy = bin_strategy
        self._memory_warn_mb = memory_warn_mb
        self._memory_limit_mb = memory_limit_mb
        self._estimate_proba_method = estimate_proba_method

        self._gum_bn = gum.BayesNet(self._name)
        self._nodes  = {}

        total_entries = memory_warn_mb * 1e6 / 8.0
        fixed_entries = 0
        n_flexible = 0
        for s in self._specs:
            d = s.dist_or_fn
            if s.n_bins_override is not None or isinstance(d, (Bernoulli, Categorical, Poisson, Binomial, _BernoulliCPT)):
                if isinstance(d, Bernoulli):
                    fixed_entries += 2
                elif isinstance(d, Categorical):
                    fixed_entries += len(d.probs)
                elif isinstance(d, (Poisson, Binomial)):
                    lo, hi = _get_discrete_domain(d)
                    fixed_entries += (hi - lo + 1)
                elif s.n_bins_override is not None and not s.parents:
                    fixed_entries += s.n_bins_override
                # _BernoulliCPT and n_bins_override with parents: skip (parent sizes unknown)
            else:
                n_flexible += 1
        budget_for_flexible = max(0.0, total_entries - fixed_entries)
        self._budget_entries_per_node = budget_for_flexible / max(1, n_flexible)

        for spec in self._specs:
            d = spec.dist_or_fn
            parents = spec.parents
            node_name = spec.name

            # Temporarily override self._n_bins for this node's compilation
            saved_n_bins = self._n_bins
            self._n_bins = self._resolve_n_bins(spec)

            lbls = spec.labels
            if callable(d) and parents:
                compiled = self._add_from_fn(node_name, d, parents, lbls)
            elif isinstance(d, _BernoulliCPT):
                compiled = self._add_bernoulli_cpt(node_name, d.cond, lbls)
            elif isinstance(d, Bernoulli):
                compiled = self._add_bernoulli_root(node_name, d.probs, lbls)
            elif isinstance(d, Categorical):
                compiled = self._add_categorical_root(node_name, d.probs, lbls)
            elif isinstance(d, (Poisson, Binomial)):
                compiled = self._add_range_root(node_name, d)
            else:
                compiled = self._add_continuous(node_name, d)

            self._n_bins = saved_n_bins   # restore

            spec.node.is_continuous = compiled.is_continuous
            spec.node.ticks = compiled.ticks

        self._evaluate_memory()
        self._state = COMPILED

    def _evaluate_memory(self) -> None:
        # Per-node CPT sizes
        node_entries: list[tuple[int, str]] = []
        for name in self._gum_bn.names():
            node_id = self._gum_bn.idFromName(name)
            var_size = self._gum_bn.variable(name).domainSize()
            parent_sz = 1
            for pid in self._gum_bn.parents(node_id):
                parent_sz *= self._gum_bn.variable(pid).domainSize()
            node_entries.append((var_size * parent_sz, name))

        total_entries = sum(e for e, _ in node_entries)
        mem_mb = total_entries * 8 / 1e6   # float64 = 8 bytes
        n_nodes = len(self._gum_bn.nodes())

        logger.info(
            "BN compiled: %d nodes, %d arcs, %s CPT entries (%.3f MB)"
            "  [n_bins=%d, strategy=%s, method=%s]",
            n_nodes, len(self._gum_bn.arcs()), f"{total_entries:,}",
            mem_mb, self._n_bins, self._bin_strategy, self._estimate_proba_method,
        )

        # Hard limit — raise before any further processing
        if self._memory_limit_mb is not None and mem_mb > self._memory_limit_mb:
            top = sorted(node_entries, reverse=True)[:3]
            top_str = ", ".join(f"{n} ({e:,} entries)" for e, n in top)
            raise RuntimeError(
                f"Compilation aborted: BN requires {mem_mb:.3f} MB "
                f"which exceeds memory_limit_mb={self._memory_limit_mb} MB.\n"
                f"Largest nodes: {top_str}\n"
                f"Suggestions:\n"
                f"- Reduce n_bins (currently {self._n_bins})\n"
                f"- Use bin_strategy=BIN_ADAPTIVE\n"
                f"- Use per-node override: bn.sample(..., n_bins=5)"
            )

        # Soft warning
        if mem_mb > self._memory_warn_mb:
            top = sorted(node_entries, reverse=True)[:3]
            top_lines = "\n".join(
                f"  {name:<30s} {entries:>8,} entries  "
                f"({len(list(self._gum_bn.parents(self._gum_bn.idFromName(name))))} parents)"
                for entries, name in top
            )
            suggestions = f"- Reduce n_bins (currently {self._n_bins})"
            if self._bin_strategy == BIN_UNIFORM:
                suggestions += "\n- Switch to bin_strategy=BIN_ADAPTIVE or BIN_MEMORY_BUDGET"
            elif self._bin_strategy == BIN_ADAPTIVE:
                suggestions += "\n- Switch to bin_strategy=BIN_MEMORY_BUDGET"
            suggestions += f"\n- Per-node override: bn.sample('{top[0][1]}', ..., n_bins=5)"
            warnings.warn(
                f"[WARN] Large BN ({mem_mb:.3f} MB > warn threshold "
                f"{self._memory_warn_mb:.3f} MB).\n"
                f"Top-3 nodes by CPT size:\n{top_lines}\n"
                f"Suggestions:\n{suggestions}",
                UserWarning,
                stacklevel=3,
            )


    def show_graph(self) -> None:
        """
        Displays the BN using graphviz (dot layout).
        Triggers compilation if still in DESIGN state.

        Requires: pip install graphviz  (+ graphviz binaries on PATH)
        """
        self._ensure_compiled()
        try:
            import graphviz
        except ImportError:
            raise ImportError(
                "graphviz package not found — pip install graphviz\n"
                "Also install the graphviz binaries: https://graphviz.org/download/"
            )

        C_DISC  = "#AED6F1"
        C_PLATE = "#FAD7A0"
        C_BOX   = "#EBF5FB"
        C_EDGE  = "#2874A6"

        # collapse plate groups: coin_1/coin_2/... → one representative node
        tmpl: dict[str, dict] = {}
        for full in self._gum_bn.names():
            tmpl[full] = {"label": full, "is_plate": full in self._plate_derived}

        plate_remap: dict[str, str] = {}
        plates_info: dict[str, dict] = {}
        for base_name, members in self._plate_groups.items():
            in_tmpl = [m for m in members if m in tmpl]
            if len(in_tmpl) > 1:
                rep = in_tmpl[0]
                for m in in_tmpl[1:]:
                    plate_remap[m] = rep
                    del tmpl[m]
                tmpl[rep]["label"] = base_name
                plates_info[base_name] = {"size": len(members), "rep": rep}

        arcs: list[tuple] = []
        seen_arcs: set = set()
        for a, b in self._gum_bn.arcs():
            s = plate_remap.get(self._gum_bn.variable(a).name(), self._gum_bn.variable(a).name())
            d = plate_remap.get(self._gum_bn.variable(b).name(), self._gum_bn.variable(b).name())
            if (s, d) not in seen_arcs and s in tmpl and d in tmpl:
                seen_arcs.add((s, d))
                arcs.append((s, d))

        g = graphviz.Digraph(name=self._name)
        g.attr(rankdir="LR", fontname="Helvetica", fontsize="11", bgcolor="white")
        g.attr("node", fontname="Helvetica", fontsize="11",
               style="filled", shape="ellipse", margin="0.15,0.08",
               penwidth="1.5", color="#2C3E50")
        g.attr("edge", arrowsize="0.8", color="#2C3E50", penwidth="1.2")

        nodes_in_cluster: set[str] = set()
        for base_name, info in plates_info.items():
            rep = info["rep"]
            nodes_in_cluster.add(rep)
            with g.subgraph(name=f"cluster_{base_name}") as c:
                c.attr(label=f"{base_name}  ×{info['size']}",
                       style="filled", fillcolor=C_BOX,
                       color=C_EDGE, fontsize="9", fontname="Helvetica",
                       fontcolor=C_EDGE)
                c.node(rep, label=tmpl[rep]["label"], fillcolor=C_PLATE)

        for n, data in tmpl.items():
            if n in nodes_in_cluster:
                continue
            g.node(n, label=data["label"], fillcolor=C_DISC)

        for s, d in arcs:
            g.edge(s, d)

        try:
            from IPython.display import display
            display(g)
        except Exception:
            g.view(cleanup=True)

    # internal methods: base nodes

    def _add_bernoulli_root(self, name: str, p: float, labels=None) -> BNNode:
        var = gum.LabelizedVariable(name, name, 2)
        var.changeLabel(0, labels[0] if labels else "False")
        var.changeLabel(1, labels[1] if labels else "True")
        self._gum_bn.add(var)
        self._gum_bn.cpt(name).fillWith([1 - p, p])
        return self._register(name, False)

    def _add_categorical_root(self, name: str, probs: list, labels=None) -> BNNode:
        var = gum.LabelizedVariable(name, name, len(probs))
        if labels:
            for i, lbl in enumerate(labels):
                var.changeLabel(i, lbl)
        self._gum_bn.add(var)
        self._gum_bn.cpt(name).fillWith(probs)
        return self._register(name, False)

    def _add_bernoulli_cpt(self, name: str, cond: _Conditional, labels=None) -> BNNode:
        var = gum.LabelizedVariable(name, name, 2)
        var.changeLabel(0, labels[0] if labels else "False")
        var.changeLabel(1, labels[1] if labels else "True")
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

    def _register(self, name: str, is_continuous: bool, ticks: Optional[np.ndarray] = None) -> BNNode:
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
                     parents: list[BNNode], labels=None) -> BNNode:
        """
        Probes dist_fn to detect the returned distribution type,
        then routes to the appropriate construction method.

            lambda p: Bernoulli(p)             -> _add_bernoulli_from_fn
            lambda p: Categorical(probs)       -> _add_categorical_from_fn
            lambda p: Normal(p, 1.0)           -> _add_continuous_conditional
        """
        probe_vals = [
            float((p.ticks[0] + p.ticks[-1]) / 2) if p.is_continuous else 0.0
            for p in parents
        ]
        probe_dist = dist_fn(*probe_vals)

        if isinstance(probe_dist, Bernoulli):
            return self._add_bernoulli_from_fn(name, dist_fn, parents, labels)
        if isinstance(probe_dist, Categorical):
            return self._add_categorical_from_fn(name, dist_fn, parents, labels)
        if isinstance(probe_dist, (Poisson, Binomial)):
            return self._add_range_from_fn(name, dist_fn, parents)
        return self._add_continuous_conditional(name, dist_fn, parents)

    def _add_bernoulli_from_fn(self, name: str, dist_fn: Callable, parents: list[BNNode], labels=None) -> BNNode:
        """Bernoulli node whose probability is a function of parents (any types)."""
        var = gum.LabelizedVariable(name, name, 2)
        var.changeLabel(0, labels[0] if labels else "False")
        var.changeLabel(1, labels[1] if labels else "True")
        self._gum_bn.add(var)
        for p in parents:
            self._gum_bn.addArc(p.name, name)

        pot  = self._gum_bn.cpt(name)
        inst = gum.Instantiation(pot)
        inst.setFirst()
        while not inst.end():
            parent_vals = [self._repr_val(inst, p) for p in parents]
            d   = dist_fn(*parent_vals)
            p_t = float(d.probs)
            x_val = inst.val(self._gum_bn.variable(name))
            pot.set(inst, p_t if x_val == 1 else 1.0 - p_t)
            inst.inc()

        return self._register(name, False)

    def _add_categorical_from_fn(self, name: str, dist_fn: Callable,parents: list[BNNode], labels=None) -> BNNode:
        """Categorical node (k values) whose probabilities are a function of parents."""
        probe_vals = [
            float((p.ticks[0] + p.ticks[-1]) / 2) if p.is_continuous else 0.0
            for p in parents
        ]
        k = len(dist_fn(*probe_vals).probs)

        var = gum.LabelizedVariable(name, name, k)
        if labels:
            for i, lbl in enumerate(labels):
                var.changeLabel(i, lbl)
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

    def _add_range_root(self, name: str, distribution) -> BNNode:
        """Root node for a discrete count distribution (Poisson / Binomial) using RangeVariable."""
        min_k, max_k = _get_discrete_domain(distribution)
        var = gum.RangeVariable(name, name, min_k, max_k)
        self._gum_bn.add(var)
        self._gum_bn.cpt(name).fillWith(_dist_to_range_cpt(distribution, min_k, max_k))
        return self._register(name, False)

    def _add_range_from_fn(self, name: str, dist_fn: Callable,
                           parents: list[BNNode]) -> BNNode:
        """Discrete count node (Poisson / Binomial) whose parameter depends on parents."""
        # Probe at parent extremes to determine the widest output domain needed
        probe_points: list[list[float]] = []
        for p in parents:
            if p.is_continuous:
                t = p.ticks
                probe_points.append([float(t[0]), float((t[0] + t[-1]) / 2), float(t[-1])])
            else:
                probe_points.append([float(i) for i in range(
                    self._gum_bn.variable(p.name).domainSize())])

        min_k, max_k = 0, 0
        for combo in iproduct(*probe_points):
            _, mk = _get_discrete_domain(dist_fn(*combo))
            max_k = max(max_k, mk)

        var = gum.RangeVariable(name, name, min_k, max_k)
        self._gum_bn.add(var)
        for p in parents:
            self._gum_bn.addArc(p.name, name)

        # Precompute PMF for every parent combination (avoids recomputing per child state)
        parent_domain = [range(self._gum_bn.variable(p.name).domainSize()) for p in parents]
        pmf_cache: dict[tuple, list[float]] = {}
        for combo in iproduct(*parent_domain):
            parent_vals = [
                float((p.ticks[idx] + p.ticks[idx + 1]) / 2) if p.is_continuous
                else float(idx)
                for p, idx in zip(parents, combo)
            ]
            pmf_cache[combo] = _dist_to_range_cpt(dist_fn(*parent_vals), min_k, max_k)

        pot  = self._gum_bn.cpt(name)
        inst = gum.Instantiation(pot)
        inst.setFirst()
        while not inst.end():
            parent_combo = tuple(inst.val(self._gum_bn.variable(p.name)) for p in parents)
            k_idx = inst.val(self._gum_bn.variable(name))
            pot.set(inst, pmf_cache[parent_combo][k_idx])
            inst.inc()

        return self._register(name, False)

    # conditional continuous node
    def _add_continuous_conditional(self, name: str, dist_fn: Callable, parents: list[BNNode]) -> BNNode:
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
                    if self._estimate_proba_method == MIDPOINT
                    else self._cpt_integration(dist_fn, combo, x_ticks)
                )

            pot.set(inst, cache[parent_bin_idxs][x_bin_idx])
            inst.inc()

        return self._register(name, True, x_ticks)

    def _cpt_midpoint(self, dist_fn: Callable, parent_combo, x_ticks: np.ndarray) -> list[float]:
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

        n_bins_x   = len(x_ticks) - 1
        cont_indices     = [i for i, info in enumerate(parent_combo) if info[3]]
        cont_indices_set = set(cont_indices)
        disc_vals  = {i: info[2] for i, info in enumerate(parent_combo)
                            if not info[3]}
        ranges     = [(parent_combo[i][0], parent_combo[i][1])
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
                cdf_hi = d.cdf(x_hi)
                cdf_lo = d.cdf(x_lo)
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
