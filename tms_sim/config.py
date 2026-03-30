"""Experiment configuration schema and JSON loader.

This module defines the configuration objects that control the simulation run.
The config is intentionally JSON-based (no external dependencies).

Key idea: peers are defined as typed entries in ``peers`` where each entry maps
to a class from :mod:`tms_sim.peers`.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from .distributions import DistSpec, parse_float_or_dist


@dataclass(frozen=True)
class CandidateConfig:
    """How many candidate sellers a buyer considers per transaction."""

    min_count: int = 5
    max_count: int = 15


@dataclass(frozen=True)
class ReceiverConfig:
    """How many buyers execute transactions at each step."""

    min_count: int = 10
    max_count: int = 10


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

        Supported modes:
        - ``mean``: arithmetic mean over trust matrix rows.
        - ``shape``: SHAPETrust algorithm.
        - ``eigen``: EigenTrust algorithm.

        For ``eigen`` mode:
        - ``alpha`` is the damping factor in ``[0, 1]``.
        - ``percentage`` is the fraction of honest peers sampled as pretrusted,
            also in ``[0, 1]``.
    """

    mode: str = "mean"  # "mean", "shape", or "eigen"
    alpha: float = 0.15
    percentage: float = 0.1


@dataclass(frozen=True)
class PeerSpecConfig:
    """A typed peer definition used to instantiate peer objects.

    Attributes:
        kind: Peer class name from ``tms_sim.peers`` (for example
            ``HonestNormalPeer`` or ``MaliciousBasicPeer``).
        count: Number of peers of this type to create.
        params: Constructor kwargs for the peer class.
        q: Optional q distribution for the base ``Peer`` kind.
        h: Optional h distribution for the base ``Peer`` kind.
    """

    kind: str
    count: int
    params: Mapping[str, Any]
    q: Optional[DistSpec] = None
    h: Optional[DistSpec] = None


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration.

    Main loop parameters:
    - ``n_steps``: number of discrete time steps.
    - ``receivers``: interval for how many buyers execute transactions at each step.

    Peer generation:
    - ``peers``: typed peer entries; total peers is the sum of their counts.
    """

    seed: int = 123
    n_steps: int = 200
    receivers: ReceiverConfig = ReceiverConfig()

    candidates: CandidateConfig = CandidateConfig()
    selection: SelectionConfig = SelectionConfig()
    price: PriceConfig = PriceConfig()
    decay: DecayConfig = DecayConfig()
    global_trust: GlobalTrustConfig = GlobalTrustConfig()

    peers: Tuple[PeerSpecConfig, ...] = ()

    @property
    def n_peers(self) -> int:
        return sum(spec.count for spec in self.peers)


def _parse_seeds(value: Any) -> list[int]:
    """Parse the `seed` field.

    Supported forms in JSON/JSON5:
    - seed: 123
    - seed: [123, 456, 789]

    Returns:
        List of integer seeds (non-empty).
    """

    if value is None:
        return [123]

    if isinstance(value, bool):
        raise ValueError("seed must be an integer or an array of integers")

    if isinstance(value, int):
        return [int(value)]

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ValueError("seed array must be non-empty")
        seeds: list[int] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError("seed array must contain only integers")
            seeds.append(int(item))
        return seeds

    raise ValueError("seed must be an integer or an array of integers")


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


def _parse_receiver_cfg(obj: Any) -> ReceiverConfig:
    """Parse receiver config from legacy int or interval object.

    Supported forms:
    - ``receivers_per_step: 10``
    - ``receivers_per_step: {min_count: 5, max_count: 12}``
    """

    if isinstance(obj, Mapping):
        min_count = int(obj.get("min_count", 10))
        max_count = int(obj.get("max_count", min_count))
        return ReceiverConfig(min_count=min_count, max_count=max_count)

    fixed = int(obj)
    return ReceiverConfig(min_count=fixed, max_count=fixed)


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
    mode = str(obj.get("mode", "mean")).lower()
    if mode not in {"mean", "shape", "eigen"}:
        raise ValueError("global_trust.mode must be one of: 'mean', 'shape', 'eigen'")

    # alpha/percentage are only relevant for eigen mode.
    if mode == "eigen":
        alpha = float(obj.get("alpha", 0.15))
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("global_trust.alpha must be in [0, 1]")

        percentage = float(obj.get("percentage", obj.get("pretrusted_percentage", 0.1)))
        if not 0.0 <= percentage <= 1.0:
            raise ValueError("global_trust.percentage must be in [0, 1]")
    elif mode == "shape":
        alpha = obj.get("alpha", 1.0)
        if alpha == "None":
            alpha = None
        else:
            alpha = float(alpha)
        percentage = 0.1
    else:
        alpha = 0.15
        percentage = 0.1

    print(f"Parsed global trust config: mode={mode}, alpha={alpha}, percentage={percentage}")
    return GlobalTrustConfig(mode=mode, alpha=alpha, percentage=percentage)


def _parse_peer_spec_cfg(obj: Mapping[str, Any]) -> PeerSpecConfig:
    """Parse one typed peer entry.

    Example:
        {"kind": "HonestNormalPeer", "count": 70}
    """

    kind = str(_require(obj, "kind"))
    count = int(obj.get("count", 1))
    params_raw = obj.get("params", {})
    if not isinstance(params_raw, Mapping):
        raise ValueError("peer params must be a JSON object")
    params = dict(params_raw)
    for k in params:
        if k in {"colluder_ids", "target_seller_ids"}:
            if isinstance(params[k], list):
                continue
            elif isinstance(params[k], str) and params[k].startswith("range(") and params[k].endswith(")"):
                range_str = params[k][len("range("):-1]
                start_str, end_str = range_str.split(",")
                start, end = int(start_str.strip()), int(end_str.strip())
                params[k] = list(range(start, end))
            else:
                raise ValueError(f"Invalid format for {k}: must be a list or range string")
            
                

    # Base Peer supports explicit q/h distribution specs.
    q = parse_float_or_dist(obj["q"]) if "q" in obj else None
    h = parse_float_or_dist(obj["h"]) if "h" in obj else None

    if kind == "Peer" and (q is None or h is None):
        raise ValueError("Peer entries require both 'q' and 'h'")

    return PeerSpecConfig(kind=kind, count=count, params=params, q=q, h=h)


def _load_experiment_raw(path: str | Path) -> Mapping[str, Any]:
    """Load the JSON/JSON5 experiment file and return the top-level mapping.

    Args:
        path: Path to a JSON or JSON5 file describing an experiment.

    Returns:
        Parsed top-level JSON object as a mapping.

    Raises:
        ValueError: If the JSON is invalid or missing required keys.
    """

    path = Path(path)
    text = path.read_text(encoding="utf-8")

    if path.suffix.lower() == ".json5":
        try:
            import json5  # type: ignore
        except ModuleNotFoundError as exc:
            raise ValueError(
                "JSON5 config requested but 'json5' package is not installed. "
                "Install it with: pip install json5"
            ) from exc
        raw = json5.loads(text)
    else:
        raw = json.loads(text)

    if not isinstance(raw, Mapping):
        raise ValueError("Experiment config must be a JSON object")

    return raw


def _build_experiment_config(raw: Mapping[str, Any], *, seed: int) -> ExperimentConfig:
    peers_raw = raw.get("peers", [])
    if not peers_raw:
        raise ValueError("peers must be provided and non-empty")

    peers = tuple(_parse_peer_spec_cfg(p) for p in peers_raw)

    return ExperimentConfig(
        seed=int(seed),
        n_steps=int(raw.get("n_steps", 200)),
        receivers=_parse_receiver_cfg(raw.get("receivers_per_step", 10)),
        candidates=_parse_candidate_cfg(raw.get("candidates", {})),
        selection=_parse_selection_cfg(raw.get("selection", {})),
        price=_parse_price_cfg(raw.get("price", {})),
        decay=_parse_decay_cfg(raw.get("decay", {})),
        global_trust=_parse_global_trust_cfg(raw.get("global_trust", {})),
        peers=peers,
    )


def load_experiment_configs(path: str | Path) -> list[ExperimentConfig]:
    """Load an experiment setup file into one or more configs.

    If `seed` is a single integer, returns a list with one config.
    If `seed` is an array, returns one config per seed.
    """

    raw = _load_experiment_raw(path)
    seeds = _parse_seeds(raw.get("seed", 123))
    return [_build_experiment_config(raw, seed=s) for s in seeds]


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment setup file into a single :class:`ExperimentConfig`.

    Note:
        If the config supplies multiple seeds (``seed`` is an array), this
        returns only the first config. Prefer :func:`load_experiment_configs`.
    """

    configs = load_experiment_configs(path)
    return configs[0]
