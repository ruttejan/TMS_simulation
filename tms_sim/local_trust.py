from dataclasses import dataclass

import math
import numpy as np

@dataclass
class TrustAccumulator:
    """Stores decayed weighted sums used for trust.

    We maintain two sums for some entity (pair buyer->seller, or seller-only):

    - N(t) = Σ w_k d_k s_k
    - D(t) = Σ w_k d_k

    Then trust is T(t) = N(t)/D(t) (or a prior when D(t)=0).
    """
    n_sum: float = 0.0
    d_sum: float = 0.0
    last_t: int = 0
    
def decay_factor(lambd: float, dt: int) -> float:
    """Compute exp(-lambda * dt) for integer time steps."""

    if dt <= 0:
        return 1.0
    return math.exp(-lambd * dt)

class LocalTrustStore:
    """Sparse store for pairwise local trust T_ij (buyer -> seller).

    This is the local trust matrix concept, but stored sparsely: only pairs that
    actually transact are allocated.
    """
    
    def __init__(self, n: int, lambd: float) -> None:
        self.lambd = lambd
        self.local_values = {}
        self.trust_matrix = np.full((n, n), np.inf) # Initialize with infinity to indicate no history


    def get_local_value(self, buyer: int, seller: int, t: int) -> float:
        """Return local trust value $T_{ij}(t)$.

        Args:
            buyer: Buyer id i.
            seller: Seller id j.
            t: Current simulation time (used to apply time decay).

        Returns:
            Local trust T_ij(t) in [0,1]. If there is no history for (i,j), returns
            the configured prior.
        """

        key = (buyer, seller)
        acc = self.local_values.get(key)
        if acc is None:
            return 0.0
        # apply decay when reading the value
        # self.apply_decay(acc, t)
        if acc.d_sum <= 0.0:
            return 0.0
        return acc.n_sum / acc.d_sum
    
    def get_matrix(self) -> np.ndarray:
        """Return the full local trust matrix T_ij."""
        return self.trust_matrix
    
    def resize_matrix(self, new_size: int) -> None:
        """Resize the local trust matrix to accommodate more peers."""
        old_size = self.trust_matrix.shape[0]
        if new_size > old_size:
            new_matrix = np.full((new_size, new_size), np.inf)
            new_matrix[:old_size, :old_size] = self.trust_matrix
            self.trust_matrix = new_matrix

    def update(self, buyer: int, seller: int, t: int, weight: float, score: float | None) -> None:
        """Add one new weighted observation to (i,j).

        Args:
            buyer: Buyer id i.
            seller: Seller id j.
            t: Current simulation time t_k.
            weight: Weight w_k (typically derived from price).
            score: Normalized score s_k in [0,1].

        Returns:
            None. (Updates internal accumulators.)
        """
        if score is None:
            return
        key = (buyer, seller)
        acc = self.local_values.get(key)
        if acc is None:
            acc = TrustAccumulator(last_t=t)
            self.local_values[key] = acc
        self.apply_decay(acc, t)
        acc.n_sum += weight * score
        acc.d_sum += weight
        
        self.trust_matrix[buyer, seller] = acc.n_sum / acc.d_sum

    def apply_decay(self, acc: TrustAccumulator, t: int) -> None:
        """Apply exponential decay from acc.last_t to time t (lazy update)."""

        dt = t - acc.last_t
        if dt <= 0:
            return
        f = decay_factor(self.lambd, dt)
        acc.n_sum *= f
        acc.d_sum *= f
        acc.last_t = t