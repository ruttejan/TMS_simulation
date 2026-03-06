"""Small utilities for sampling and numerical helpers.

This module keeps config parsing independent of any heavy dependencies.
It provides:

- :class:`DistSpec` for sampling peer parameters (q, h) from simple distributions
- :func:`safe_softmax` for stable softmax sampling of sellers
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Mapping, Optional, Protocol, Union


class RandomLike(Protocol):
    """Protocol for RNG objects used by :class:`DistSpec`.

    Python's :class:`random.Random` satisfies this protocol.
    """

    def random(self) -> float: ...
    def betavariate(self, alpha: float, beta: float) -> float: ...
    def uniform(self, a: float, b: float) -> float: ...


@dataclass(frozen=True)
class DistSpec:
    """A distribution specification parsed from config.

    Supported:
    - fixed: {"dist": "fixed", "value": 0.7}
    - uniform: {"dist": "uniform", "low": 0.2, "high": 0.9}
    - beta: {"dist": "beta", "a": 2.0, "b": 5.0}
    """

    dist: str
    params: Mapping[str, Any]

    def sample(self, rng: RandomLike) -> float:
        """Sample one value from the configured distribution."""

        d = self.dist.lower()
        if d == "fixed":
            return float(self.params["value"])
        if d == "uniform":
            return float(rng.uniform(float(self.params["low"]), float(self.params["high"])))
        if d == "beta":
            return float(rng.betavariate(float(self.params["a"]), float(self.params["b"])))
        raise ValueError(f"Unsupported dist: {self.dist!r}")


def parse_float_or_dist(value: Union[float, int, Mapping[str, Any]]) -> DistSpec:
    """Parse either a numeric constant or a dict distribution spec."""

    if isinstance(value, (float, int)):
        return DistSpec("fixed", {"value": float(value)})
    if isinstance(value, Mapping):
        if "dist" not in value:
            raise ValueError("Distribution spec must include 'dist'")
        dist = str(value["dist"])
        params = dict(value)
        params.pop("dist", None)
        return DistSpec(dist, params)
    raise TypeError(f"Expected number or mapping, got {type(value).__name__}")


def clamp01(x: float) -> float:
    """Clamp to the [0,1] interval."""

    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def safe_softmax(scores: Mapping[int, float], beta: float, rng: random.Random) -> int:
    """Sample one key using a numerically-stable softmax.

    Args:
        scores: mapping candidate_id -> score
        beta: inverse temperature; higher means more greedy
        rng: RNG

    Returns:
        The sampled key.
    """

    if not scores:
        raise ValueError("scores empty")
    scaled = {k: beta * v for k, v in scores.items()}
    m = max(scaled.values())
    weights = {k: math.exp(v - m) for k, v in scaled.items()}
    total = sum(weights.values())
    if total <= 0 or not math.isfinite(total):
        # Fallback to argmax if something went numerically wrong.
        return max(scores, key=scores.get)
    r = rng.random() * total
    acc = 0.0
    for k, w in weights.items():
        acc += w
        if r <= acc:
            return k
    # numerical edge
    return next(iter(weights))
