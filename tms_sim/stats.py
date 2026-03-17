"""Aggregate statistical measurements for an experiment.

This module maintains running totals so you can report values like:

- success rate based on ground truth outcomes o_k
- average reported stars / normalized score
- good-pick rate (how often a buyer selects a "good" seller by q threshold)

Note: the current Transaction model does not expose explicit honesty indicator
``z_honest``, so dishonesty rate cannot be measured directly.
"""

from __future__ import annotations

from dataclasses import dataclass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .peers import Peer

from .transaction import Transaction

def plot_global_trust(peers: list[Peer], global_values: list[float], filename: str) -> None:
    """Plot global trust values over time and save to file.

    Each peer type is assigned a distinct color and marker shape.
    """
    from .peers import (
        HonestNormalPeer, HonestSupremePeer,
        MaliciousBasicPeer, MaliciousRaterPeer, FreeRiderPeer,
        TargetingMaliciousRaterPeer, TraitorPeer,
        CollusiveBasicPeer, CollusiveTargetingPeer, SybilAccountPeer,
    )

    # color, marker per peer class
    _STYLE: dict[type, tuple[str, str]] = {
        HonestNormalPeer:           ("tab:blue",   "o"),
        HonestSupremePeer:          ("tab:cyan",   "^"),
        MaliciousBasicPeer:         ("tab:red",    "X"),
        MaliciousRaterPeer:         ("tab:orange", "s"),
        FreeRiderPeer:              ("tab:gray",   "D"),
        TargetingMaliciousRaterPeer:("tab:brown",  "P"),
        TraitorPeer:                ("tab:pink",   "v"),
        CollusiveBasicPeer:         ("tab:purple", "p"),
        CollusiveTargetingPeer:     ("tab:olive",  "h"),
        SybilAccountPeer:           ("black",      "*"),
    }
    _DEFAULT_STYLE = ("tab:green", "o")

    # Group peer indices by type so each group gets one legend entry.
    from collections import defaultdict
    groups: dict[type, list[int]] = defaultdict(list)
    for i, peer in enumerate(peers):
        groups[type(peer)].append(i)

    fig, ax = plt.subplots(figsize=(9, 5))
    for peer_cls, indices in groups.items():
        color, marker = _STYLE.get(peer_cls, _DEFAULT_STYLE)
        xs = indices
        ys = [global_values[i] for i in indices]
        ax.scatter(xs, ys, c=color, marker=marker, label=peer_cls.__name__, zorder=3)

    ax.set_xlabel("Peer ID")
    ax.set_ylabel("Global Trust Value")
    ax.set_title("Global Trust Values of Peers")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


@dataclass
class Stats:
    """Streaming (online) aggregation of experiment metrics."""

    total: int = 0
    succ: int = 0
    # Kept for backward compatibility in snapshot output.
    dishonest: int = 0
    sum_r: float = 0.0
    sum_s: float = 0.0
    good_picks: int = 0

    def update(self, tx: Transaction, seller_q: float, q_min_good: float) -> None:
        """Update aggregate counters with one new transaction.

        Args:
            tx: The transaction record.
            seller_q: Ground-truth seller quality q_s (used only for GoodPickRate).
            q_min_good: Threshold for classifying a seller as "good".

        Returns:
            None. (Updates internal counters.)
        """

        self.total += 1
        self.succ += int(tx.outcome_ok)
        if tx.rating is not None:
            self.sum_r += float(tx.rating)
        if tx.s_norm is not None:
            self.sum_s += float(tx.s_norm)
        if seller_q >= q_min_good:
            self.good_picks += 1
        if int(tx.outcome_ok) == 1 and tx.rating is not None and tx.rating < 3:
            self.dishonest += 1

    def snapshot(self) -> dict:
        """Return a JSON-serializable dict of current metric values.

        Returns:
            Dict with keys:
            - total
            - succ_rate
            - dishon_rate
            - avg_rating
            - avg_score
            - good_pick_rate
        """

        if self.total <= 0:
            return {
                "total": 0,
                "succ_rate": 0.0,
                "dishon_rate": 0.0,
                "avg_rating": 0.0,
                "avg_score": 0.0,
                "good_pick_rate": 0.0,
            }
        return {
            "total": self.total,
            "succ_rate": self.succ / self.total,
            "dishon_rate": self.dishonest / self.total,
            "avg_rating": self.sum_r / self.total,
            "avg_score": self.sum_s / self.total,
            "good_pick_rate": self.good_picks / self.total,
        }
