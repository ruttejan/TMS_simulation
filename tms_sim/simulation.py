import random
from pathlib import Path
from dataclasses import dataclass
from typing import Any
import pandas as pd


from .config import ExperimentConfig
from .distributions import clamp01
from .price import PriceHandler
from .selection import SellerSelection
from .stats import Stats, plot_global_trust
from .peers import *
from .transaction import Transaction, evaluate_transaction
from .local_trust import LocalTrustStore
from .global_trust import GlobalTrustStore, SHAPETrustStore, EigenTrustStore


########################################################
# SUPPORT METHODS
########################################################


def _build_peers(cfg: ExperimentConfig, rng: random.Random) -> tuple[list[Peer], list[int], list[int], list[int]]:
    """Create peer instances from the configured typed peer specs."""

    peer_types: dict[str, type[Peer]] = {
        "Peer": Peer,
        "HonestNormalPeer": HonestNormalPeer,
        "HonestSupremePeer": HonestSupremePeer,
        "MaliciousBasicPeer": MaliciousBasicPeer,
        "MaliciousRaterPeer": MaliciousRaterPeer,
        "FreeRiderPeer": FreeRiderPeer,
        "FreeRiderBuyerPeer": FreeRiderBuyerPeer,
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
            if spec_q is None:
                raise ValueError("Peer entries require q")
            q = clamp01(spec_q.sample(rng))
            return Peer(peer_id=peer_id, q=q, rng=rng)

        ctor = peer_types[kind]
        return ctor(peer_id=peer_id, rng=rng, **params)

    peers: list[Peer] = []
    collusive_peer_ids = []
    sybil_accounts_ids = []
    freeriding_buyer_ids = []
    peer_id = 0
    for spec in cfg.peers:
        params = dict(spec.params)
        for _ in range(spec.count):
            peers.append(_make_peer(spec.kind, peer_id, params, spec.q, spec.h))
            if spec.kind in ("CollusiveBasicPeer", "CollusiveTargetingPeer"):
                collusive_peer_ids.append(peer_id)
            if spec.kind == "SybilAccountPeer":
                sybil_accounts_ids.append(peer_id)
            if spec.kind == "FreeRiderBuyerPeer":
                freeriding_buyer_ids.append(peer_id)
            peer_id += 1
            
    return peers, collusive_peer_ids, sybil_accounts_ids, freeriding_buyer_ids

def sample_peer_ids(rng:random.Random, n: int, blacklist: list[int], low: int, high: int) -> list[int]:
    """Sample peer ids from 0 to n-1 excluding blacklist."""
    candidates = [i for i in range(n) if i not in blacklist]
    k_low = max(1, low)
    k_high = max(k_low, high)
    k = min(rng.randint(k_low, k_high), len(candidates))
    return rng.sample(candidates, k=k)

@dataclass
class ExperimentResult:
    """Outputs from a single experiment run."""

    config: ExperimentConfig
    stats: dict
    transactions: list[Transaction]
    
def create_global_trust_store(cfg: ExperimentConfig, n: int, peers: list[Peer], rng: random.Random) -> GlobalTrustStore:
    """Factory for global trust store based on config."""
    options = cfg.global_trust.options

    if cfg.global_trust.mode == "mean":
        return GlobalTrustStore(n=n)
    elif cfg.global_trust.mode == "shape":
        tau = options.get("tau", None)
        if tau == "None":
            tau = None
        elif tau is not None:
            tau = float(tau)
        return SHAPETrustStore(n=n, tau=tau)
    elif cfg.global_trust.mode == "eigen":
        # select pretrusted peers randomly from honest peers (with percentage specified in config)
        honest_peers = [peer for peer in peers if isinstance(peer, (HonestNormalPeer, HonestSupremePeer))]
        percentage = float(options.get("percentage", options.get("pretrusted_percentage", 0.05)))
        if not 0.0 <= percentage <= 1.0:
            raise ValueError("global_trust percentage must be in [0, 1]")

        alpha = float(options.get("alpha", 0.15))
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("global_trust alpha must be in [0, 1]")

        n_pretrusted = int(len(honest_peers) * percentage)
        n_pretrusted = max(2, min(n_pretrusted, len(honest_peers)))
        pretrusted = rng.sample(honest_peers, k=n_pretrusted) if n_pretrusted > 0 else []
        pretrusted_ids = [peer.peer_id for peer in pretrusted]
        return EigenTrustStore(n=n, pretrusted=pretrusted_ids, alpha=alpha)
    else:
        raise ValueError(f"Unknown global trust mode: {cfg.global_trust.mode!r}")
    
def check_global_trust_store_vs_normalization(global_trust: GlobalTrustStore, normalization_mode: str) -> bool:
    """
    Check if the global trust store is compatible with the normalization mode.
    
    EigenTrustStore needs to be used with "negative" normalization mode, because it produces values in the range [-1, 1].
    """
    if isinstance(global_trust, EigenTrustStore) and normalization_mode != "negative":
        return False
    return True

########################################################
# END OF SUPPORT METHODS
########################################################


def run_experiment(cfg: ExperimentConfig, *, plot_path: str | Path | None = None) -> ExperimentResult:
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

    peers, collusive_peer_ids, sybil_accounts_ids, freeriding_buyer_ids = _build_peers(cfg, rng)
    
    n = len(peers)
    if n <= 1:
        raise ValueError("Need at least 2 peers")

    # Initialize components.
    price_handler = PriceHandler(mu=cfg.price.mu, sigma=cfg.price.sigma)

    local_trust = LocalTrustStore(n=n, lambd=cfg.decay.lambd)
    
    global_trust = create_global_trust_store(cfg, n=n, peers=peers, rng=rng)

    selector = SellerSelection(mode=cfg.selection.mode, theta=cfg.selection.theta, beta=cfg.selection.beta)
    
    if not check_global_trust_store_vs_normalization(global_trust, cfg.normalization.mode):
        normalization_mode = "negative"  # Force compatible normalization mode
    else:
        normalization_mode = cfg.normalization.mode
    
    stats = Stats(n=n)
    stats.reset()
    transactions: list[Transaction] = []
    
    # delete sybil accounts from all_peers to prevent them from being selected as buyers or sellers in the main loop
    all_peers = list(range(n))
    for peer_id in sybil_accounts_ids:
        all_peers.remove(peer_id)
        
    # sybil cannot be regular buyers
    buyers_blacklist = sybil_accounts_ids
    # sybil and freeriding buyers cannot be regular sellers
    sellers_blacklist = sybil_accounts_ids + freeriding_buyer_ids 

    for t in range(1, cfg.n_steps + 1):
        # Show progress every 10%
        if t % max(1, cfg.n_steps // 10) == 0:
            progress = (t / cfg.n_steps) * 100
            print(f"\rProgress: {progress:.1f}%", end="", flush=True)
        
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
                        rng=rng,
                        norm=normalization_mode
                    )
                    local_trust.update(buyer=peer_id, seller=colluder_id, t=t, weight=tx.price_weight, score=tx.s_norm)
                    stats.update_collusive()
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
                    rng=rng,
                    norm=normalization_mode
                )
                local_trust.update(buyer=peer_id, seller=main_account_id, t=t, weight=tx.price_weight, score=tx.s_norm)
                stats.update_collusive()
        
        # Sample how many buyers act this step from the configured interval.
        receivers = sample_peer_ids(rng, n, buyers_blacklist, cfg.receivers.min_count, cfg.receivers.max_count)
        
        for buyer in receivers:
            
            candidates = sample_peer_ids(rng, n, sellers_blacklist, cfg.candidates.min_count, cfg.candidates.max_count)
            
            # Ensure buyer is not in candidates
            while buyer in candidates:
                del candidates[candidates.index(buyer)]
                candidates.append(sample_peer_ids(rng, n, sellers_blacklist + candidates, low=1, high=1)[0])

            # Gather local and global trust for the candidates
            lt = {j: local_trust.get_local_value(buyer, j, t=t) for j in candidates}
            gt = {j: global_trust.get_global_value(j) for j in candidates}

            # Select the seller
            seller = None
            while seller is None and candidates != []:
                seller = selector.select(buyer=buyer, candidates=candidates, local_trust=lt, global_trust=gt, rng=rng)

            if candidates == [] or seller is None:
                continue
            
            stats.update_pick(peers[buyer], peers[seller], [peers[j] for j in candidates])
            
            # Simulate transaction
            tx = evaluate_transaction(
                buyer=peers[buyer],
                seller=peers[seller],
                t=t,
                price_handler=price_handler,
                rng=rng,
                norm=normalization_mode
            )

            # Update local trust stores
            local_trust.update(buyer=buyer, seller=seller, t=t, weight=tx.price_weight, score=tx.s_norm)

            # Aggregate statistics (for plotting / experiment comparison)
            stats.update_normal(tx)
            transactions.append(tx)
            
        # update global trust values
        global_trust.update(local_trust.get_matrix())
    print()  # New line after progress bar completes
        
    # get the output folder path from plot_path if provided, otherwise use current directory
    output_folder = Path(plot_path).parent if plot_path is not None else Path(".")
    # save final global trust values to csv
    gt_df = pd.DataFrame({
        "global_trust": global_trust.global_values,
    })
    
    gt_df.to_csv(f"{output_folder}/global_trust.csv", index=False)

    if plot_path is None:
        plot_file = f"{cfg.global_trust.mode}_seed_{cfg.seed}.png"
    else:
        plot_file = str(Path(plot_path))
    plot_global_trust(peers, global_values=global_trust.global_values, filename=plot_file)
    stats.plot_seller_buyer_distribution(filename=plot_file.replace(".png", "_distribution.png"))
    
    return ExperimentResult(config=cfg, stats=stats.snapshot(), transactions=transactions)
