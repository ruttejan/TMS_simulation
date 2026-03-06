"""Experiment configuration schema and JSON loader.

This module defines the configuration objects that control the simulation run.
The config is intentionally JSON-based (no external dependencies).

Key idea: peers are generated from one or more *peer groups*. Each peer has two
parameters:

- ``q``: service quality (seller success probability)
- ``h``: reporting honesty (buyer truthfulness probability)

Distributions for ``q`` and ``h`` can be specified either as a fixed number or as
a small distribution spec (see :class:`tms_sim.distributions.DistSpec`).
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .distributions import DistSpec, parse_float_or_dist


@dataclass(frozen=True)
class CandidateConfig:
    """How many candidate sellers a buyer considers per transaction."""

    min_count: int = 5
    max_count: int = 15


@dataclass(frozen=True)
class SelectionConfig:
    """Seller selection policy.

    - ``mode`` chooses greedy argmax vs probabilistic softmax.
    - ``alpha`` mixes local vs global trust: score = alpha*T_ij + (1-alpha)*G_j.
    - ``beta`` is softmax inverse temperature (larger => more greedy).
    """

    mode: str = "softmax"  # "softmax" or "argmax"
    alpha: float = 0.7
    beta: float = 8.0


@dataclass(frozen=True)
class PriceConfig:
    """Transaction price generation + weighting parameters."""

    mu: float = 0.0
    sigma: float = 1.0
    r_max: float = 10.0


@dataclass(frozen=True)
class DecayConfig:
    """Exponential time decay for trust evidence: d = exp(-lambda * Δt)."""

    lambd: float = 0.02  # time decay lambda


@dataclass(frozen=True)
class GlobalTrustConfig:
    """How global trust is represented in this implementation.

    The code maintains a *seller-level* reputation store (decayed weighted average of
    observed normalized scores). The snapshot can either be used directly ("mean")
    or normalized to sum to 1 ("normalized").
    """

    mode: str = "mean"  # "mean" in [0,1] or "normalized" sums to 1
    prior: float = 0.5


@dataclass(frozen=True)
class PeerGroupConfig:
    """A group of peers with shared parameter distributions."""

    name: str
    count: int
    q: DistSpec
    h: DistSpec


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration.

    Main loop parameters:
    - ``n_steps``: number of discrete time steps.
    - ``receivers_per_step``: how many buyers execute transactions at each step.

    Peer generation:
    - ``peer_groups``: list of groups; total peers is the sum of their counts.
    """

    seed: int = 123
    n_steps: int = 200
    receivers_per_step: int = 10

    candidates: CandidateConfig = CandidateConfig()
    selection: SelectionConfig = SelectionConfig()
    price: PriceConfig = PriceConfig()
    decay: DecayConfig = DecayConfig()
    global_trust: GlobalTrustConfig = GlobalTrustConfig()

    q_min_good: float = 0.7

    peer_groups: Tuple[PeerGroupConfig, ...] = ()

    @property
    def n_peers(self) -> int:
        return sum(g.count for g in self.peer_groups)


def _require(mapping: Mapping[str, Any], key: str) -> Any:
    """Fetch a required key from a JSON object with a friendly error."""
    if key not in mapping:
        raise ValueError(f"Missing required key: {key}")
    return mapping[key]


def _parse_candidate_cfg(obj: Mapping[str, Any]) -> CandidateConfig:
    return CandidateConfig(
        min_count=int(obj.get("min_count", 5)),
        max_count=int(obj.get("max_count", 15)),
    )


def _parse_selection_cfg(obj: Mapping[str, Any]) -> SelectionConfig:
    return SelectionConfig(
        mode=str(obj.get("mode", "softmax")),
        alpha=float(obj.get("alpha", 0.7)),
        beta=float(obj.get("beta", 8.0)),
    )


def _parse_price_cfg(obj: Mapping[str, Any]) -> PriceConfig:
    return PriceConfig(
        mu=float(obj.get("mu", 0.0)),
        sigma=float(obj.get("sigma", 1.0)),
        r_max=float(obj.get("r_max", 10.0)),
    )


def _parse_decay_cfg(obj: Mapping[str, Any]) -> DecayConfig:
    """Parse time decay config.

    Accepts either ``{"lambda": ...}`` or ``{"lambd": ...}``.
    """

    # JSON can't use "lambda" comfortably in some editors, so accept both.
    if "lambda" in obj and "lambd" in obj:
        raise ValueError("Use only one of 'lambda' or 'lambd'")
    lambd = obj.get("lambd", obj.get("lambda", 0.02))
    return DecayConfig(lambd=float(lambd))


def _parse_global_trust_cfg(obj: Mapping[str, Any]) -> GlobalTrustConfig:
    return GlobalTrustConfig(
        mode=str(obj.get("mode", "mean")),
        prior=float(obj.get("prior", 0.5)),
    )


def _parse_peer_group_cfg(obj: Mapping[str, Any]) -> PeerGroupConfig:
    """Parse one peer group entry.

    Example:
        {"name": "honest", "count": 50, "q": 0.9, "h": 1.0}
    """

    name = str(_require(obj, "name"))
    count = int(_require(obj, "count"))
    q = parse_float_or_dist(_require(obj, "q"))
    h = parse_float_or_dist(_require(obj, "h"))
    return PeerGroupConfig(name=name, count=count, q=q, h=h)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment setup JSON into an :class:`ExperimentConfig`.

    Args:
        path: Path to a JSON file describing an experiment.

    Returns:
        Parsed :class:`ExperimentConfig` instance.

    Raises:
        ValueError: If the JSON is invalid or missing required keys.
    """

    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Experiment config must be a JSON object")

    peer_groups_raw = raw.get("peer_groups", [])
    if not peer_groups_raw:
        raise ValueError("peer_groups must be provided and non-empty")

    peer_groups = tuple(_parse_peer_group_cfg(g) for g in peer_groups_raw)

    return ExperimentConfig(
        seed=int(raw.get("seed", 123)),
        n_steps=int(raw.get("n_steps", 200)),
        receivers_per_step=int(raw.get("receivers_per_step", 10)),
        candidates=_parse_candidate_cfg(raw.get("candidates", {})),
        selection=_parse_selection_cfg(raw.get("selection", {})),
        price=_parse_price_cfg(raw.get("price", {})),
        decay=_parse_decay_cfg(raw.get("decay", {})),
        global_trust=_parse_global_trust_cfg(raw.get("global_trust", {})),
        q_min_good=float(raw.get("q_min_good", 0.7)),
        peer_groups=peer_groups,
    )
