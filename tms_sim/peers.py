from dataclasses import dataclass
import random

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

class Peer:
    """Represents a peer in the TMS simulation."""
    
    def __init__(self, peer_id: int, q: float, rng: random.Random):
        self.peer_id = peer_id
        self.q = q  # intrinsic quality (probability of successful transaction)
        self.rng = rng
        self.is_honest = True
    
    def sample_outcome(self, price_weight: float, t: int) -> int:
        """Sample a transaction outcome (0 or 1) based on the peer's quality q."""
        return 1 if self.rng.random() < self.q else 0
    
    def sample_stars(self, outcome_ok: int, seller_id: int) -> int|None:
        """Sample stars based on the transaction outcome."""
        dist = SUCCESS_STAR_DIST if outcome_ok == 1 else FAIL_STAR_DIST
        rating = _sample_discrete(dist, self.rng)
        return rating
    
    def __str__(self):
        return "DefaultPeer"

class HonestNormalPeer(Peer):
    """Always honest peer with good quality."""
    
    def __init__(self, peer_id: int, rng: random.Random):
        # q sampled from uniform(0.9,0.95)
        q = rng.uniform(0.9, 0.95)
        super().__init__(peer_id, q=q, rng=rng)
        self.is_honest = True
        
    def __str__(self):
        return "HonestNormal"

class HonestSupremePeer(Peer):
    """Always honest peer with very good quality."""
    
    def __init__(self, peer_id: int, rng: random.Random):
        q = 1.0
        super().__init__(peer_id, q=q, rng=rng)
        self.is_honest = True
        
    def __str__(self):
        return "HonestSupreme"

class MaliciousBasicPeer(Peer):
    """Peer with bad quality and 3 modes of honesty (honest, inverse, bad_rater)"""
    
    def __init__(self, peer_id: int, rng: random.Random, type: str = "bad_rater"):
        # q sampled from uniform(0,0.2)
        q = rng.uniform(0, 0.1)
        super().__init__(peer_id, q=q, rng=rng)
        self.type = type
        self.is_honest = False
        
    def sample_stars(self, outcome_ok: int, seller_id: int) -> int|None:
        if self.type == "honest":
            dist = SUCCESS_STAR_DIST if outcome_ok == 1 else FAIL_STAR_DIST
            rating = _sample_discrete(dist, self.rng)
            return rating
        elif self.type == "inverse":
            dist = FAIL_STAR_DIST if outcome_ok == 1 else SUCCESS_STAR_DIST
            rating = _sample_discrete(dist, self.rng)
            return rating
        elif self.type == "bad_rater":
            dist = FAIL_STAR_DIST
            rating = _sample_discrete(dist, self.rng)
            return rating
        else:
            raise ValueError(f"Unknown malicious peer type: {self.type!r}")
        
    def __str__(self):
        return f"MaliciousBasic({self.type})"
        
class MaliciousRaterPeer(Peer):
    """Always bad rating peer with good quality"""
    
    def __init__(self, peer_id: int, rng: random.Random):
        # q sampled from uniform(0.8, 0.9)
        q = rng.uniform(0.9, 0.95)
        super().__init__(peer_id, q=q, rng=rng)
        self.is_honest = False
        
    def sample_stars(self, outcome_ok: int, seller_id: int) -> int|None:
        dist = FAIL_STAR_DIST
        rating = _sample_discrete(dist, self.rng)
        return rating
    
    def __str__(self):
        return "MaliciousRater"
    
class FreeRiderPeer(Peer):
    """Peer with good quality that provides no ratings"""
    
    def __init__(self, peer_id: int, rng: random.Random):
        # q sampled from uniform(0.8, 0.9)
        q = rng.uniform(0.9, 0.95)
        super().__init__(peer_id, q=q, rng=rng)
        self.is_honest = True
        
    def sample_stars(self, outcome_ok: int, seller_id: int) -> int|None:
        return None  # does not provide any rating
    
    def __str__(self):
        return "FreeRider"
    
class FreeRiderBuyerPeer(Peer):
    """Peer that only buys and rates but does not sell. It provides honest ratings.
    The "not selling" is implemented in the simulation logic.
    """
    
    def __init__(self, peer_id: int, rng: random.Random):
        # q sampled from uniform(0.8, 0.9)
        q = rng.uniform(0.9, 0.95)
        super().__init__(peer_id, q=q, rng=rng)
        self.is_honest = True
    
    def __str__(self):
        return "FreeRiderBuyer"

class TargetingMaliciousRaterPeer(Peer):
    """Peer with good quality that provides bad ratings only 
        to a specific set of target sellers (e.g., competitors).
    """
    
    def __init__(self, peer_id: int, rng: random.Random, target_seller_ids: list[int]):
        # q sampled from uniform(0.8, 0.9)
        q = rng.uniform(0.9, 0.95)
        super().__init__(peer_id, q=q, rng=rng)
        self.target_seller_ids = target_seller_ids
        self.is_honest = False

    def sample_stars(self, outcome_ok: int, seller_id: int) -> int|None:
        if seller_id in self.target_seller_ids:
            dist = FAIL_STAR_DIST
        else:
            dist = SUCCESS_STAR_DIST if outcome_ok == 1 else FAIL_STAR_DIST
        rating = _sample_discrete(dist, self.rng)
        return rating
    
    def __str__(self):
        return f"TargetingMaliciousRater"

class TraitorPeer(Peer):
    """Peer that behaves honestly for a while and then turns malicious."""
    
    def __init__(self, peer_id: int, rng: random.Random, betrayal_gap: int = 10, periodic: bool = True):
        # q sampled from uniform(0.9, 0.95)
        q = rng.uniform(0.9, 0.95)
        super().__init__(peer_id, q=q, rng=rng)
        self.betrayal_gap = betrayal_gap
        self.number_of_honest_sells = 0
        self.periodic = periodic
        self.is_honest = False
        
    def sample_outcome(self, price_weight: float, t: int) -> int:
        """Traitor can cheat only on big transaction 
        or on every transaction with a periodic pattern (e.g., every 10th transaction)."""
        if self.periodic:
            if self.number_of_honest_sells >= self.betrayal_gap:
                self.number_of_honest_sells = 0
                return 0
            else:
                self.number_of_honest_sells += 1
                return super().sample_outcome(price_weight, t)
        else:
            if price_weight > 1.5:
                return 0
            else:
                return super().sample_outcome(price_weight, t)
            
    def __str__(self):
        return f"Traitor({"periodic" if self.periodic else "price-based"})"
    
class CollusiveBasicPeer(Peer):
    """Peer with either bad or good quality that provides 
        perfect rating for colluders and bad rating for others.
    """
    
    def __init__(self, peer_id: int, rng: random.Random, colluder_ids: list[int], quality=False):
        # q sampled from uniform(0.9, 0.95)
        q = rng.uniform(0.9, 0.95) if quality else rng.uniform(0, 0.1)
        super().__init__(peer_id, q=q, rng=rng)
        self.colluder_ids = colluder_ids
        self.is_honest = False
        
    def sample_stars(self, outcome_ok: int, seller_id: int) -> int|None:
        if seller_id in self.colluder_ids:
            rating = 5
        else:
            dist = FAIL_STAR_DIST
            rating = _sample_discrete(dist, self.rng)
        return rating
    
    def __str__(self):
        return f"CollusiveBasic"
    
class CollusiveTargetingPeer(Peer):
    """Peer with either bad or good quality that provides 
        perfect rating for colluders and bad rating for others, 
        but only rates a subset of sellers (e.g., popular ones).
    """
    
    def __init__(self, peer_id: int, rng: random.Random, colluder_ids: list[int], target_seller_ids: list[int], quality=False):
        # q sampled from uniform(0.9, 0.95)
        q = rng.uniform(0.9, 0.95) if quality else rng.uniform(0, 0.1)
        super().__init__(peer_id, q=q, rng=rng)
        self.colluder_ids = colluder_ids
        self.target_seller_ids = target_seller_ids
        self.is_honest = False

    def sample_stars(self, outcome_ok: int, seller_id: int) -> int|None:
        if seller_id in self.colluder_ids:
            rating = 5
        elif seller_id not in self.target_seller_ids:
            dist = SUCCESS_STAR_DIST if outcome_ok == 1 else FAIL_STAR_DIST
            rating = _sample_discrete(dist, self.rng)
        else:
            dist = FAIL_STAR_DIST
            rating = _sample_discrete(dist, self.rng)
        return rating
    
    def __str__(self):
        return f"CollusiveTargeting"
    
class SybilAccountPeer(Peer):
    """Accounts created by a malicious to boost his rating. 
        They only rate the main account with perfect rating and do not transact with others.
        The main account can be any previously defined self-operating peer 
        (e.g., malicious rater, traitor, etc.) that the sybil accounts are trying to boost.
    """
    
    def __init__(self, peer_id: int, rng: random.Random, main_account_id: int):
        q = 0.0
        super().__init__(peer_id, q=q, rng=rng)
        self.main_account_id = main_account_id
        self.is_honest = False
        
    def sample_stars(self, outcome_ok: int, seller_id: int) -> int|None:
        if seller_id == self.main_account_id:
            rating = 5
        else:
            rating = None  # does not transact with others (for implementation purposes)
        return rating
    
    def __str__(self):
        return f"SybilAccount"
    