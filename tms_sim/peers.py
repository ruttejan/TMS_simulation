
from dataclasses import dataclass

@dataclass(frozen=True)
class Peer:
    """Represents a peer in the TMS simulation."""
    
    peer_id: int
    q: float  # intrinsic quality (probability of successful transaction)
    h: float  # honesty (probability of reporting true outcome)