"""Aggregate statistical measurements for an experiment.

This module maintains running totals so you can report values like:

- success rate based on ground truth outcomes o_k
- average reported stars / normalized score
- good-pick rate (how often a buyer selects an honest seller)

Note: the current Transaction model does not expose explicit honesty indicator
``z_honest``, so dishonesty rate cannot be measured directly.
"""

from __future__ import annotations

from dataclasses import dataclass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .peers import *
import numpy as np

from .transaction import Transaction

def plot_global_trust(peers: list[Peer], global_values: list[float] | np.ndarray, filename: str) -> None:
    """Plot global trust values over time and save to file.

    Each peer type is assigned a distinct color and marker shape.
    """
    

    # color, marker per peer class
    _STYLE: dict[type, tuple[str, str]] = {
        HonestNormalPeer:           ("tab:blue",   "o"),
        HonestSupremePeer:          ("tab:cyan",   "^"),
        MaliciousBasicPeer:         ("tab:red",    "X"),
        MaliciousRaterPeer:         ("tab:orange", "s"),
        FreeRiderPeer:              ("tab:gray",   "D"),
        FreeRiderBuyerPeer:         ("tab:gray",   "d"),
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
        
    # get sybil main account index and plot a circle around it
    sybil_main_accounts = set([peer.main_account_id for peer in peers if isinstance(peer, SybilAccountPeer)])
    for main_account_index in sybil_main_accounts:
        ax.scatter(main_account_index, global_values[main_account_index], facecolors="none", edgecolors="black", s=100, linewidths=1.5, zorder=4, label="Sybil Main Account")
            
    ax.set_xlabel("Peer ID")
    ax.set_ylabel("Global Trust Value")
    ax.set_title("Global Trust Values of Peers")
    ax.legend(loc="best", fontsize=8)
    ax.set_xticks([x for x in range(len(peers)) if x % 10 == 0])
    ax.grid(True, linestyle="--", alpha=0.5, zorder=0)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


class Stats:
    """Streaming (online) aggregation of experiment metrics."""

    total_normal: int = 0
    total_collusive: int = 0
    succ: int = 0
    good_pick: int = 0
    bad_pick: int = 0
    bad_pick_dict: dict[str, int] = {}
    other_pick: int = 0
    # Kept for backward compatibility in snapshot output.
    dishonest: int = 0
    sum_r: float = 0.0
    sum_s: float = 0.0
    

    def update_normal(self, tx: Transaction) -> None:
        """Update aggregate counters with one new transaction.

        Args:
            tx: The transaction record.

        Returns:
            None. (Updates internal counters.)
        """

        self.total_normal += 1
        self.succ += int(tx.outcome_ok)
        if tx.rating is not None:
            self.sum_r += float(tx.rating)
        if tx.s_norm is not None:
            self.sum_s += float(tx.s_norm)
        if int(tx.outcome_ok) == 1 and tx.rating is not None and tx.rating < 3:
            self.dishonest += 1
            
    def update_collusive(self) -> None:
        """ Update collusive transaction counter"""
        self.total_collusive += 1
        
    def update_pick(self, buyer: Peer, pick: Peer, candidates: list[Peer]) -> None:
        """Update pick counters based on the selected peer and candidate list."""
        if np.all([peer.is_honest for peer in candidates]) or (np.all([not peer.is_honest for peer in candidates])) or buyer.is_honest == False:
            # All candidates are honest or all candidates are dishonest, so we cannot classify the pick as good or bad.
            self.other_pick += 1
        elif pick.is_honest:
            self.good_pick += 1
        else:
            self.bad_pick += 1
            pick_type = type(pick).__name__
            self.bad_pick_dict[pick_type] = self.bad_pick_dict.get(pick_type, 0) + 1

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
        return {
            "total_normal": self.total_normal,
            "total_collusive": self.total_collusive,
            "succ_rate": self.succ / self.total_normal if self.total_normal > 0 else 0.0,
            "dishon_rate": self.dishonest / self.total_normal if self.total_normal > 0 else 0.0,
            "avg_rating": self.sum_r / self.total_normal if self.total_normal > 0 else 0.0,
            "avg_score": self.sum_s / self.total_normal if self.total_normal > 0 else 0.0,
            "good_pick_rate": self.good_pick / self.total_normal if self.total_normal > 0 else 0.0,
            "bad_pick_rate": self.bad_pick / self.total_normal if self.total_normal > 0 else 0.0,
            "other_pick_rate": self.other_pick / self.total_normal if self.total_normal > 0 else 0.0,
            "bad_pick_distribution": {k: v / self.bad_pick if self.bad_pick > 0 else 0.0 for k, v in self.bad_pick_dict.items()},
        }
        
    def reset(self) -> None:
        """Reset all counters to zero."""
        self.total_normal = 0
        self.total_collusive = 0
        self.succ = 0
        self.dishonest = 0
        self.sum_r = 0.0
        self.sum_s = 0.0
        self.good_pick = 0
        self.bad_pick = 0
        self.other_pick = 0
        self.bad_pick_dict = {}