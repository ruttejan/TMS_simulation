"""Main simulation loop.

Implements the non-Appendix parts of `ideas/simulation_overview.md`:

- At each time step, sample a set of buyers (receivers)
- For each buyer, sample candidate sellers and select one using a mixed score
- Simulate the transaction: outcome -> truthful stars -> (possibly) inverted report
- Normalize stars to [0,1]
- Apply price-based weight and exponential time decay
- Update local trust T_ij and global seller reputation G_j
- Update aggregate statistics

Important simplification: peers are always online (no churn).
"""

import math
import random
from dataclasses import dataclass

from .config import ExperimentConfig
from .distributions import clamp01
from .price import PriceHandler
from .selection import SellerSelection
# from .stats import Stats
# from .transaction import TransactionModel
from .peers import Peer
from .transaction import Transaction
from .local_trust import LocalTrustStore
from .global_trust import GlobalTrustStore, SHAPETrustStore, EigenTrustStore


def _build_peers(cfg: ExperimentConfig, rng: random.Random) -> list[Peer]:
    """Create peer instances from the configured peer groups.

    Each peer i gets:
    - q_i sampled from the group's q distribution
    - h_i sampled from the group's h distribution

    Args:
        cfg: Experiment configuration (defines peer groups).
        rng: Random number generator used for sampling.

    Returns:
        List of :class:`Peer` objects with assigned peer_id, q, and h.
    """

    peers: list[Peer] = []
    peer_id = 0
    for group in cfg.peer_groups:
        for _ in range(group.count):
            q = clamp01(group.q.sample(rng))
            h = clamp01(group.h.sample(rng))
            peers.append(Peer(peer_id=peer_id, q=q, h=h))
            peer_id += 1
    return peers


@dataclass
class ExperimentResult:
    """Outputs from a single experiment run."""

    config: ExperimentConfig
    stats: dict
    transactions: list[Transaction]


def run_experiment(cfg: ExperimentConfig) -> ExperimentResult:
    """Run a complete experiment.

    Args:
        cfg: Experiment configuration.

    Returns:
        :class:`ExperimentResult` containing:
        - the config
        - a JSON-serializable statistics dict
        - a list of all simulated :class:`Transaction` records

    Raises:
        ValueError: If there are fewer than 2 peers.
    """

    rng = random.Random(cfg.seed)

    peers = _build_peers(cfg, rng)
    
    n = len(peers)
    if n <= 1:
        raise ValueError("Need at least 2 peers")

    # Initialize components.
    price_handler = PriceHandler(r_max=cfg.price.r_max)

    local_trust = LocalTrustStore(n=n, lambd=cfg.decay.lambd)
    global_trust = GlobalTrustStore(n=n)

    tx_model = TransactionModel(mu=cfg.price.mu, sigma=cfg.price.sigma)
    selector = SellerSelection(mode=cfg.selection.mode, alpha=cfg.selection.alpha, beta=cfg.selection.beta)
    
    stats = Stats()
    transactions: list[Transaction] = []

    for t in range(1, cfg.n_steps + 1):
        # Peers are always online, so every peer is eligible each step.
        all_peers = list(range(n))
        k = min(cfg.receivers_per_step, n)
        receivers = rng.sample(all_peers, k=k)

        for buyer in receivers:
            # Candidate sellers: sample a random subset excluding buyer
            available = [pid for pid in all_peers if pid != buyer]
            if not available:
                continue

            m_low = max(1, cfg.candidates.min_count)
            m_high = max(m_low, cfg.candidates.max_count)
            m = rng.randint(m_low, m_high)
            m = min(m, len(available))
            candidates = rng.sample(available, k=m)

            # Gather local trust for these candidates (buyer-specific)
            lt = {j: local_trust.get_local_value(buyer, j, t=t) for j in candidates}
            gt = {j: global_trust.get_global_value(j) for j in candidates}

            # Select the seller
            seller = selector.select(buyer=buyer, candidates=candidates, local_trust=lt, global_trust=gt, rng=rng)

            # Simulate transaction
            tx = tx_model.simulate(
                buyer=buyer,
                buyer_h=peers[buyer].h,
                seller=seller,
                seller_q=peers[seller].q,
                t=t,
                rng=rng,
            )

            # Get price weight before updating the price handler with the new price
            w = price_handler.weight_from_price(tx.price)
            price_handler.update_mean(tx.price)

            # Update local trust stores
            local_trust.update(buyer=buyer, seller=seller, t=t, weight=w, score=tx.s_norm)

            # Aggregate statistics (for plotting / experiment comparison)
            stats.update(tx, seller_q=peers[seller].q, q_min_good=cfg.q_min_good)
            transactions.append(tx)
            
        # update global trust values
        global_trust.update(local_trust.get_matrix())

    return ExperimentResult(config=cfg, stats=stats.snapshot(), transactions=transactions)
