"""Seller selection policy.

Given a buyer i and a set of candidate sellers, compute a combined score:

    score_ij = alpha * T_ij + (1 - alpha) * G_j

and choose a seller either greedily (argmax) or probabilistically (softmax).
"""

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping

from .distributions import safe_softmax


@dataclass(frozen=True)
class SellerSelection:
    """Implements argmax/softmax seller selection on mixed local/global trust."""

    mode: str  # "softmax" or "argmax"
    alpha: float
    beta: float

    def select(self, buyer: int, candidates: List[int], local_trust: Mapping[int, float], global_trust: Mapping[int, float], rng: random.Random) -> int:
        """Select a seller from candidates.

        Args:
            buyer: Buyer peer id (not used directly in scoring, but included for clarity).
            candidates: List of candidate seller ids.
            local_trust: Mapping seller_id -> T_ij (buyer-specific local trust values).
            global_trust: Mapping seller_id -> G_j (global reputation values).
            rng: Random number generator.

        Returns:
            The selected seller id.

        Notes:
            The combined score is:
                score_ij = alpha * T_ij + (1 - alpha) * G_j
        """

        if not candidates:
            raise ValueError("candidates empty")

        scores: Dict[int, float] = {}
        for j in candidates:
            tij = float(local_trust.get(j, 0.5))
            gj = float(global_trust.get(j, 0.5))
            scores[j] = self.alpha * tij + (1.0 - self.alpha) * gj

        m = self.mode.lower()
        if m == "argmax":
            return max(scores, key=scores.get) # type: ignore[return-value]
        if m == "softmax":
            return safe_softmax(scores, beta=self.beta, rng=rng)
        raise ValueError(f"Unknown selection mode: {self.mode!r}")