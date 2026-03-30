"""Seller selection policy.

Given a buyer i and a set of candidate sellers, compute a combined score:

    score_ij = alpha * T_ij + (1 - alpha) * G_j

and choose a seller either greedily (argmax) or probabilistically (softmax).
"""

import random
import numpy as np
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping
from .local_trust import LocalTrustStore
from .global_trust import GlobalTrustStore

from .distributions import safe_softmax


@dataclass(frozen=True)
class SellerSelection:
    """Implements argmax/softmax seller selection on mixed local/global trust."""

    mode: str  # "softmax" or "argmax"
    alpha: float
    beta: float

    def select(self, 
               buyer: int, 
               candidates: List[int], 
               local_trust: Mapping[int, float], 
               global_trust: Mapping[int, float], 
               rng: random.Random) -> int:
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
            # print(f"Debug: buyer={buyer}, seller={j}, T_ij={tij:.3f}, G_j={gj:.3f}")
            scores[j] = self.alpha * tij + (1.0 - self.alpha) * gj

        m = self.mode.lower()
        if m == "argmax":
            return max(scores, key=scores.get) # type: ignore[return-value]
        if m == "softmax":
            return safe_softmax(scores, beta=self.beta, rng=rng)
        raise ValueError(f"Unknown selection mode: {self.mode!r}")
    
    def reject(self, 
               seller: int, 
               buyer: int, 
               t: int,
               local_trust: LocalTrustStore, 
               global_trust: GlobalTrustStore,
               reject_threshold: float = 0.1) -> bool:
        """Determine whether to reject a seller based on local/global trust."""
        tij = float(local_trust.get_local_value(seller, buyer, t))
        gj = float(global_trust.get_global_value(buyer))
        buyer_score = self.alpha * tij + (1.0 - self.alpha) * gj
        
        # get average score of all known peers of the seller
        known_peers_local = local_trust.get_row(seller)
        known_peers_global = global_trust.global_values
        if known_peers_local.shape != known_peers_global.shape:
            raise ValueError("Local and global trust vectors must have the same shape")
        known_mask = known_peers_local != np.inf
        known_peers_local = known_peers_local[known_mask] # filter out unknown peers
        known_peers_global = known_peers_global[known_mask] # filter out unknown peers
        if len(known_peers_local) < 10: # if there are less than 10 known peers, we cannot reliably compute an average, so we do not reject
            return False
        # avg_peer_score = np.median(self.alpha * known_peers_local + (1.0 - self.alpha) * known_peers_global)    
        # if buyer_score < avg_peer_score * reject_threshold: # if the buyer's score is less than the specified threshold of the average peer score, we reject
        #     return True
        # return False
        if tij != 0:
            peer_scores = self.alpha * known_peers_local + (1.0 - self.alpha) * known_peers_global
            iqr = np.percentile(peer_scores, 75) - np.percentile(peer_scores, 25)
            lower_bound = np.percentile(peer_scores, 25) - 1.5 * iqr
            if buyer_score < lower_bound: # if the buyer's score is less
                return True
            return False
        else:
            peer_scores = known_peers_global
            iqr = np.percentile(peer_scores, 75) - np.percentile(peer_scores, 25)
            lower_bound = np.percentile(peer_scores, 25) - 1.5 * iqr
            if buyer_score < lower_bound: # if the buyer's score is less
                return True
            return False