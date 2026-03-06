
from dataclasses import dataclass


@dataclass(frozen=True)
class Transaction:
    """A single simulated transaction (one buyer–seller interaction)."""

    t: int
    buyer: int
    seller: int
    outcome_ok: int  # o_k in {0,1}
    r_true: int
    z_honest: int  # z_k in {0,1}
    r_reported: int  # r_k in {0..5}
    s_norm: float  # r_k/5
    price: float
    price_weight: float