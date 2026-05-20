from dataclasses import dataclass, field
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
    - ``theta`` mixes local vs global trust: score = theta*T_ij + (1-theta)*G_j.
    - ``beta`` is softmax inverse temperature (larger => more greedy).
    """

    mode: str = "softmax"  # "softmax" or "argmax"
    theta: float = 0.7
    beta: float = 8.0


@dataclass(frozen=True)
class PriceConfig:
    """Transaction price generation model: log-normal distribution with parameters mu and sigma."""

    mu: float = 0.0
    sigma: float = 0.6


@dataclass(frozen=True)
class DecayConfig:
    """Exponential time decay for trust evidence: d = lambda^(-Δt)."""

    lambd: float = 1.02  # time decay lambda


@dataclass(frozen=True)
class GlobalTrustConfig:
    """How global trust is represented in this implementation.

    Supported modes:
    - ``mean``: arithmetic mean over trust matrix rows.
    - ``shape``: SHAPETrust algorithm.
    - ``eigen``: EigenTrust algorithm.

    Algorithm-specific options are stored in ``options`` and passed through to
    the selected trust implementation.
    """

    mode: str = "mean"  # "mean", "shape", or "eigen"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizationConfig:
    """How global trust values are normalized.

    Supported modes:
    - ``positive``: Normalize to the interval [0, 1].
    - ``negative``: Normalize to the interval [-1, 1].
    """

    mode: str = "negative"  # "positive" or "negative"


@dataclass(frozen=True)
class PeerSpecConfig:
    """A typed peer definition used to instantiate peer objects.

    Attributes:
        kind: Peer class name from ``tms_sim.peers`` (for example
            ``HonestNormalPeer`` or ``MaliciousBasicPeer``).
        count: Number of peers of this type to create.
        params: Constructor kwargs for the peer class.
        q: Optional q distribution for the base ``Peer`` kind.
    """

    kind: str
    count: int
    params: Mapping[str, Any]
    q: Optional[DistSpec] = None


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration."""

    seed: int = 123
    n_steps: int = 200
    receivers: ReceiverConfig = ReceiverConfig()

    candidates: CandidateConfig = CandidateConfig()
    selection: SelectionConfig = SelectionConfig()
    price: PriceConfig = PriceConfig()
    decay: DecayConfig = DecayConfig()
    global_trust: GlobalTrustConfig = GlobalTrustConfig()
    normalization: NormalizationConfig = NormalizationConfig()

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
        print(f"Parsed seeds: {seeds}")
        return seeds

    raise ValueError("seed must be an integer or an array of integers")


def _require(mapping: Mapping[str, Any], key: str) -> Any:
    """Fetch a required key from a JSON object with a friendly error."""
    if key not in mapping:
        raise ValueError(f"Missing required key: {key}")
    return mapping[key]


def _parse_candidate_cfg(obj: Any) -> CandidateConfig:
    """Parse candidate config from legacy int or interval object.
    
    Supported forms:
    - ``candidates: 10``
    - ``candidates: {min_count: 5, max_count: 15}``
    """
    
    if isinstance(obj, Mapping):
        return CandidateConfig(
            min_count=int(obj.get("min_count", 5)),
            max_count=int(obj.get("max_count", 15)),
        )
        
    fixed = int(obj)
    return CandidateConfig(min_count=fixed, max_count=fixed)

def _parse_receiver_cfg(obj: Any) -> ReceiverConfig:
    """Parse receiver config from legacy int or interval object.

    Supported forms:
    - ``receivers_per_step: 10``
    - ``receivers_per_step: {min_count: 5, max_count: 12}``
    """

    if isinstance(obj, Mapping):
        min_count = int(obj.get("min_count", 10))
        max_count = int(obj.get("max_count", 20))
        return ReceiverConfig(min_count=min_count, max_count=max_count)

    fixed = int(obj)
    return ReceiverConfig(min_count=fixed, max_count=fixed)


def _parse_selection_cfg(obj: Mapping[str, Any]) -> SelectionConfig:
    return SelectionConfig(
        mode=str(obj.get("mode", "softmax")),
        theta=float(obj.get("theta", 0.7)),
        beta=float(obj.get("beta", 8.0)),
    )


def _parse_price_cfg(obj: Mapping[str, Any]) -> PriceConfig:
    return PriceConfig(
        mu=float(obj.get("mu", 0.0)),
        sigma=float(obj.get("sigma", 0.6)),
    )


def _parse_decay_cfg(obj: Mapping[str, Any]) -> DecayConfig:
    lambd =  obj.get("lambda", 1.02)
    return DecayConfig(lambd=float(lambd))


def _parse_global_trust_cfg(obj: Mapping[str, Any]) -> GlobalTrustConfig:
    mode = str(obj.get("mode", "mean")).lower()
    if mode not in {"mean", "shape", "eigen"}:
        raise ValueError("global_trust.mode must be one of: 'mean', 'shape', 'eigen'")

    options = {key: value for key, value in obj.items() if key != "mode"}
    return GlobalTrustConfig(mode=mode, options=options)


def _parse_normalization_cfg(obj: Mapping[str, Any]) -> NormalizationConfig:
    """Parse normalization config.

    Supported options:
    - ``positive``: Normalization to the interval [0, 1]
    - ``negative``: Normalization to the interval [-1, 1]
    """

    mode = str(obj.get("mode", "negative")).lower()
    if mode not in {"positive", "negative"}:
        raise ValueError("normalization.mode must be one of: 'positive', 'negative'")

    return NormalizationConfig(mode=mode)


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

    if kind == "Peer" and (q is None):
        raise ValueError("Peer entries require 'q'")

    return PeerSpecConfig(kind=kind, count=count, params=params, q=q)


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
        normalization=_parse_normalization_cfg(raw.get("normalization", {})),
        peers=peers,
    )


def load_experiment_configs(path: str | Path) -> list[ExperimentConfig]:
    """Load an experiment setup file into one or more configs.

    If `seed` is a single integer, returns a list with one config.
    If `seed` is an array, returns one config per seed.
    """

    raw = _load_experiment_raw(path)
    seeds = _parse_seeds(raw.get("seed", 123))

    # Allow specifying multiple global trust algorithms to run in one config file.
    # Supported forms:
    # - global_trust.mode: single string (legacy)
    # - global_trust.modes: array of strings, e.g. ["mean", "shape", "eigen"]
    # - global_trust.modes: array of objects for per-mode overrides, e.g.
    #     [{mode: "shape", alpha: "None"}, {mode: "eigen", alpha: 0.15}]
    gt_raw = raw.get("global_trust", {}) if isinstance(raw, Mapping) else {}

    modes_list: list = []
    if isinstance(gt_raw, Mapping):
        if "modes" in gt_raw:
            raw_modes = gt_raw.get("modes")
        elif "mode" in gt_raw:
            raw_modes = gt_raw.get("mode")
        else:
            raw_modes = None

        if raw_modes is None:
            modes_list = []
        elif isinstance(raw_modes, (list, tuple)):
            modes_list = list(raw_modes)
        else:
            # single string or single mapping
            modes_list = [raw_modes]

    # Normalize to at least one entry, default to mean
    if not modes_list:
        modes_list = ["mean"]

    configs: list[ExperimentConfig] = []
    for s in seeds:
        for mode_entry in modes_list:
            raw_for_mode = dict(raw)
            base_gt = dict(raw_for_mode.get("global_trust", {}))

            if isinstance(mode_entry, Mapping):
                if "mode" not in mode_entry:
                    raise ValueError("Each mode object in global_trust.modes must include a 'mode' key")
                mode_name = str(mode_entry["mode"]).lower()
                gt_copy = dict(base_gt)
                # Merge overrides from the mode entry (strings/numbers as provided)
                for k, v in mode_entry.items():
                    gt_copy[k] = v
                gt_copy["mode"] = mode_name
            else:
                mode_name = str(mode_entry).lower()
                gt_copy = dict(base_gt)
                gt_copy["mode"] = mode_name

            raw_for_mode["global_trust"] = gt_copy
            configs.append(_build_experiment_config(raw_for_mode, seed=s))

    return configs


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment setup file into a single :class:`ExperimentConfig`.

    Note:
        If the config supplies multiple seeds (``seed`` is an array), this
        returns only the first config. Prefer :func:`load_experiment_configs`.
    """

    configs = load_experiment_configs(path)
    return configs[0]
