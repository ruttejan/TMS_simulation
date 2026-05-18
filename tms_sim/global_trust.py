import numpy as np
from .eigentrust import eigentrust
from .shapetrust import shapetrust_numba

class GlobalTrustStore:
    """Maintains a vector of global trust values for all sellers."""
    
    def __init__(self, n: int):
        self.n = n
        self.global_values = np.zeros(n)  # Initialize global trust values to 0 for all sellers
        
    def get_global_value(self, seller: int) -> float:
        """Return global trust value G_j for a seller."""
        return self.global_values[seller]
    
    def update(self, trust_matrix: np.ndarray) -> None:
        """Update global trust values based on the trust matrix - basic average method"""
        # filter out np.inf values
        trust_matrix_filtered = np.nan_to_num(trust_matrix, posinf=0.0)
        # Average the trust values for each seller
        counts = np.sum(trust_matrix != np.inf, axis=0)
        self.global_values = np.divide(
            np.sum(trust_matrix_filtered, axis=0),
            counts,
            out=np.zeros_like(counts, dtype=float),
            where=counts > 0,
        )

            
class SHAPETrustStore (GlobalTrustStore):
    """Maintains a vector of global trust values for all sellers using SHAPE-Trust method."""
    def __init__(self, n: int, tau = None):
        super().__init__(n)
        self.tau = tau  # Optional parameter for weighting internal vs external value, can be adjusted as needed
        self.calculate_tau = tau is None  # Flag to determine if tau should be calculated dynamically
        self.internal_values = np.zeros(n)  # Store internal trust values for debugging/analysis
        self.external_values = np.zeros(n)  # Store external trust values for debugging/analysis

    def normalize_global_values(self):
        """Normalize global trust values to be between -1 and 1."""
        min_val = abs(np.min(self.global_values))
        max_val = abs(np.max(self.global_values))
        divisor = max(max_val, min_val) + 1e-10  # Avoid division by zero
        self.global_values = self.global_values / divisor
        
    def update(self, trust_matrix: np.ndarray) -> None:
        """Update global trust values based on the trust matrix using SHAPE method."""
        self.internal_values, self.external_values = shapetrust_numba(trust_matrix)
        if self.calculate_tau:
            # calculate the tau
            self.tau = - np.sum(np.abs(self.internal_values)) / (np.sum(np.abs(self.external_values)) + 1e-10)  # Avoid division by zero

        # Combine internal and external values (can be weighted if tau is set)
        if self.tau is None:
            self.tau = 1.0
        
        self.global_values = self.internal_values + self.tau * self.external_values
        self.normalize_global_values()

                
class EigenTrustStore (GlobalTrustStore):
    """Maintains a vector of global trust values for all sellers using EigenTrust method."""
    def __init__(self, n: int, pretrusted: list[int] | np.ndarray = [], alpha: float = 0.15):
        super().__init__(n)
        self.pretrusted = pretrusted
        self.alpha = alpha  # Damping factor for EigenTrust, can be adjusted as needed
    
    def update(self, trust_matrix: np.ndarray) -> None:
        """Update global trust values based on the trust matrix using EigenTrust method."""
        # change np.inf to 0 in trust_matrix before calling eigentrust
        trust_matrix = np.nan_to_num(trust_matrix, posinf=0.0)
        self.global_values = eigentrust(trust_matrix, pretrusted=self.pretrusted, alpha=self.alpha, eps=1e-10, max_iter=100_000)