
from dataclasses import dataclass
import random

from .price import PriceHandler
from .peers import Peer

@dataclass(frozen=True)
class Transaction:
    """A single simulated transaction (one buyer–seller interaction)."""

    t: int
    buyer: int
    seller: int
    outcome_ok: int  # o_k in {0,1}
    rating: int | None
    s_norm: float | None # r_k/5
    price: float
    price_weight: float
    

SUCCESS_STAR_DIST = [(5, 0.6), (4, 0.3), (3, 0.1)]
FAIL_STAR_DIST = [(0, 0.5), (1, 0.3), (2, 0.2)]

def _sample_discrete(dist: list[tuple[int, float]], rng: random.Random) -> int:
    """Sample from a small discrete distribution given as (value, probability)."""

    r = rng.random()
    acc = 0.0
    for value, p in dist:
        acc += p
        if r <= acc:
            return value
    return dist[-1][0]
    
def evaluate_transaction(buyer: Peer, seller: Peer, t: int, price_handler: PriceHandler, rng: random.Random) -> Transaction:
    """Simulate one transaction record.
    
    Args:
        buyer: Buyer peer object (with h parameter).
        seller: Seller peer object (with q parameter).
        t: Discrete simulation time t_k.
        price: Transaction price p_k.
    Returns:
        A :class:`Transaction` with fields:
        - outcome_ok: o_k ∈ {0,1}
        - r_true: truthful stars
        - z_honest: honesty indicator z_k ∈ {0,1}
        - r_reported: reported stars after inversion (if dishonest)
        - s_norm: normalized score in [0,1]
        - price: transaction price
    """
    
    # Price draw
    price = price_handler.gen_price(rng)
    price_weight = price_handler.weight_from_price(price)
    # Get price weight before updating the price handler with the new price
    price_handler.update_mean(price)
    
    # Outcome generation
    outcome_ok = seller.sample_outcome(price_weight=price_weight, t=t)

    # Rating generation
    rating = buyer.sample_stars(outcome_ok, buyer_id=buyer.peer_id) 

    # Normalized score
    s_norm = None if rating is None else rating / 5.0
    

    return Transaction(
        t=t,
        buyer=buyer.peer_id,
        seller=seller.peer_id,
        outcome_ok=outcome_ok,
        rating=rating,
        s_norm=float(s_norm) if s_norm is not None else None,
        price=float(price),
        price_weight=float(price_weight)
    )

    