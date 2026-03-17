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
from typing import Any

from .config import ExperimentConfig
from .distributions import clamp01
from .price import PriceHandler
from .selection import SellerSelection
from .stats import Stats, plot_global_trust
from .peers import *
from .transaction import Transaction, evaluate_transaction
from .local_trust import LocalTrustStore
from .global_trust import GlobalTrustStore, SHAPETrustStore, EigenTrustStore


def _build_peers(cfg: ExperimentConfig, rng: random.Random) -> tuple[list[Peer], list[int], list[int]]:
    """Create peer instances from the configured typed peer specs."""

    peer_types: dict[str, type[Peer]] = {
        "Peer": Peer,
        "HonestNormalPeer": HonestNormalPeer,
        "HonestSupremePeer": HonestSupremePeer,
        "MaliciousBasicPeer": MaliciousBasicPeer,
        "MaliciousRaterPeer": MaliciousRaterPeer,
        "FreeRiderPeer": FreeRiderPeer,
        "TargetingMaliciousRaterPeer": TargetingMaliciousRaterPeer,
        "TraitorPeer": TraitorPeer,
        "CollusiveBasicPeer": CollusiveBasicPeer,
        "CollusiveTargetingPeer": CollusiveTargetingPeer,
        "SybilAccountPeer": SybilAccountPeer,
    }

    def _make_peer(kind: str, peer_id: int, params: dict[str, Any], spec_q: Any, spec_h: Any) -> Peer:
        if kind not in peer_types:
            raise ValueError(f"Unknown peer kind: {kind!r}")

        if kind == "Peer":
            if spec_q is None or spec_h is None:
                raise ValueError("Peer entries require both q and h")
            q = clamp01(spec_q.sample(rng))
            h = clamp01(spec_h.sample(rng))
            return Peer(peer_id=peer_id, q=q, h=h, rng=rng)

        ctor = peer_types[kind]
        return ctor(peer_id=peer_id, rng=rng, **params)

    peers: list[Peer] = []
    collusive_peer_ids = []
    sybil_accounts_ids = []
    peer_id = 0
    for spec in cfg.peers:
        params = dict(spec.params)
        for _ in range(spec.count):
            peers.append(_make_peer(spec.kind, peer_id, params, spec.q, spec.h))
            if spec.kind in ("CollusiveBasicPeer", "CollusiveTargetingPeer"):
                collusive_peer_ids.append(peer_id)
            if spec.kind == "SybilAccountPeer":
                sybil_accounts_ids.append(peer_id)
            peer_id += 1
    return peers, collusive_peer_ids, sybil_accounts_ids

def sample_peer_ids(rng:random.Random, all_peer_ids: list[int], low: int, high: int) -> list[int]:
    """Sample peer ids from all_peer_ids excluding exclude_ids."""
    k_low = max(1, low)
    k_high = max(k_low, high)
    k = min(rng.randint(k_low, k_high), len(all_peer_ids))
    return rng.sample(all_peer_ids, k=k)

@dataclass
class ExperimentResult:
    """Outputs from a single experiment run."""

    config: ExperimentConfig
    stats: dict
    transactions: list[Transaction]
    
def create_global_trust_store(cfg: ExperimentConfig, n: int, peers: list[Peer], rng: random.Random) -> GlobalTrustStore:
    """Factory for global trust store based on config."""
    if cfg.global_trust.mode == "mean":
        return GlobalTrustStore(n=n)
    elif cfg.global_trust.mode == "shape":
        return SHAPETrustStore(n=n)
    elif cfg.global_trust.mode == "eigen":
        # select pretrusted peers randomly from honest peers (with percentage specified in config)
        honest_peers = [peer for peer in peers if isinstance(peer, (HonestNormalPeer, HonestSupremePeer))]
        n_pretrusted = int(len(honest_peers) * cfg.global_trust.percentage)
        n_pretrusted = max(0, min(n_pretrusted, len(honest_peers)))
        pretrusted = rng.sample(honest_peers, k=n_pretrusted) if n_pretrusted > 0 else []
        pretrusted_ids = [peer.peer_id for peer in pretrusted]
        alpha = cfg.global_trust.alpha
        return EigenTrustStore(n=n, pretrusted=pretrusted_ids, alpha=alpha)
    else:
        raise ValueError(f"Unknown global trust mode: {cfg.global_trust.mode!r}")


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

    peers, collusive_peer_ids, sybil_accounts_ids = _build_peers(cfg, rng)
    
    n = len(peers)
    if n <= 1:
        raise ValueError("Need at least 2 peers")

    # Initialize components.
    price_handler = PriceHandler(r_max=cfg.price.r_max, mu=cfg.price.mu, sigma=cfg.price.sigma)

    local_trust = LocalTrustStore(n=n, lambd=cfg.decay.lambd)
    
    global_trust = create_global_trust_store(cfg, n=n, peers=peers, rng=rng)

    selector = SellerSelection(mode=cfg.selection.mode, alpha=cfg.selection.alpha, beta=cfg.selection.beta)
    
    stats = Stats()
    transactions: list[Transaction] = []
    
    # delete sybil accounts from all_peers to prevent them from being selected as buyers or sellers in the main loop
    all_peers = list(range(n))
    for peer_id in sybil_accounts_ids:
        all_peers.remove(peer_id)

    for t in range(1, cfg.n_steps + 1):
        
        # Make transactions between colluders and between sybil accounts 
        # and their main account.
        if t % 10 == 0:
            for peer_id in collusive_peer_ids:
                peer = peers[peer_id]
                if not isinstance(peer, (CollusiveBasicPeer, CollusiveTargetingPeer)):
                    continue
                for colluder_id in peer.colluder_ids:
                    tx = evaluate_transaction(
                        buyer=peers[peer_id],
                        seller=peers[colluder_id],
                        t=t,
                        price_handler=price_handler,
                        rng=rng
                    )
                    stats.update(tx, seller_q=peers[colluder_id].q, q_min_good=cfg.q_min_good)
                    transactions.append(tx)
            for peer_id in sybil_accounts_ids:
                peer = peers[peer_id]
                if not isinstance(peer, SybilAccountPeer):
                    continue
                main_account_id = peer.main_account_id
                tx = evaluate_transaction(
                    buyer=peers[peer_id],
                    seller=peers[main_account_id],
                    t=t,
                    price_handler=price_handler,
                    rng=rng
                )
                stats.update(tx, seller_q=peers[main_account_id].q, q_min_good=cfg.q_min_good)
                transactions.append(tx)
        
        # Sample how many buyers act this step from the configured interval.
        receivers = sample_peer_ids(rng, all_peers, cfg.receivers.min_count, cfg.receivers.max_count)

        for buyer in receivers:
            
            candidates = sample_peer_ids(rng, all_peers, cfg.candidates.min_count, cfg.candidates.max_count)
            
            # Ensure buyer is not in candidates
            while buyer in candidates:
                del candidates[candidates.index(buyer)]
                candidates.append(rng.sample(all_peers, k=1)[0])

            # Gather local and global trust for the candidates
            lt = {j: local_trust.get_local_value(buyer, j, t=t) for j in candidates}
            gt = {j: global_trust.get_global_value(j) for j in candidates}

            # Select the seller
            seller = None
            while seller is None:
                seller = selector.select(buyer=buyer, candidates=candidates, local_trust=lt, global_trust=gt, rng=rng)
                # TODO: implement seller-side acceptance policy (e.g., based on price or trust)
                    # For now, we assume the selected seller always accepts the transaction.
                # if peers[seller].accept_transaction(buyer=buyer):
                #     del candidates[candidates.index(seller)]  # remove selected seller from candidates for next iteration
                #     seller = None  # reset seller to trigger re-selection
            
            # Simulate transaction
            tx = evaluate_transaction(
                buyer=peers[buyer],
                seller=peers[seller],
                t=t,
                price_handler=price_handler,
                rng=rng
            )

            # Update local trust stores
            local_trust.update(buyer=buyer, seller=seller, t=t, weight=tx.price_weight, score=tx.s_norm)

            # Aggregate statistics (for plotting / experiment comparison)
            stats.update(tx, seller_q=peers[seller].q, q_min_good=cfg.q_min_good)
            transactions.append(tx)
            
        # update global trust values
        global_trust.update(local_trust.get_matrix())

    print(f"Final local trust matrix:\n{local_trust.get_matrix()}")
    # Plot global trust values
    print(f"Final global trust values: {global_trust.global_values}")
    plot_global_trust(peers, global_values=global_trust.global_values, filename=f"global_trust_{cfg.global_trust.mode}.png")
    return ExperimentResult(config=cfg, stats=stats.snapshot(), transactions=transactions)
