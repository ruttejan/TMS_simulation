import numpy as np
from .eigentrust import eigentrust
from .shapetrust import shapetrust, shapetrust_numba

class GlobalTrustStore:
    """Maintains a vector of global trust values for all sellers."""
    
    def __init__(self, n: int):
        self.n = n
        self.global_values = np.zeros(n)  # Initialize global trust values to 0 for all sellers
        
    def get_global_value(self, seller: int) -> float:
        """Return global trust value G_j for a seller."""
        return self.global_values[seller]
    
    def update(self, trust_matrix: np.ndarray) -> None:
        """Update global trust values based on the trust matrix."""
        # filter out np.inf values in trust_matrix before computing mean
        trust_matrix = np.nan_to_num(trust_matrix, posinf=0.0)
        self.global_values = np.sum(trust_matrix, axis=0)
        
    def resize_vector(self, new_size: int) -> None:
        """Resize the global trust vector to accommodate more sellers."""
        if new_size > self.n:
            new_global_values = np.zeros(new_size)
            new_global_values[:self.n] = self.global_values
            self.global_values = new_global_values
            self.n = new_size
            
class SHAPETrustStore (GlobalTrustStore):
    """Maintains a vector of global trust values for all sellers using SHAPE method."""
    def __init__(self, n: int, alpha = None):
        super().__init__(n)
        self.alpha = alpha  # Optional parameter for weighting internal vs external value, can be adjusted as needed
        self.calculate_alpha = alpha is None  # Flag to determine if alpha should be calculated dynamically
        
    def normalize_global_values(self) -> None:
        """Normalize global trust values to be between 0 and 1."""
        min_val = np.min(self.global_values)
        if min_val < 0: # shift values to be non-negative
            self.global_values -= min_val
        # normalize so they sum to 1
        total = np.sum(self.global_values)
        if total > 0:
            self.global_values /= total
            
        
    def update(self, trust_matrix: np.ndarray) -> None:
        """Update global trust values based on the trust matrix using SHAPE method."""
        internal, external = shapetrust_numba(trust_matrix, max=True)
        if self.calculate_alpha:
            # calculate the alpha
            self.alpha = - np.sum(np.abs(internal)) / (np.sum(np.abs(external)) + 1e-10)  # Avoid division by zero
        
        # Combine internal and external values (can be weighted if alpha is set)
        # self.global_values = internal + self.alpha * external 
        self.global_values = -external
        # self.normalize_global_values()
        
class EigenTrustStore (GlobalTrustStore):
    """Maintains a vector of global trust values for all sellers using EigenTrust method."""
    def __init__(self, n: int, pretrusted: list[int] | np.ndarray = [], alpha: float = 0.15):
        super().__init__(n)
        self.pretrusted = pretrusted
        self.alpha = alpha  # Damping factor for EigenTrust, can be adjusted as needed
    
    def update(self, trust_matrix: np.ndarray) -> None:
        """Update global trust values based on the trust matrix using EigenTrust method."""
        # Implement calling of the EigenTrust algorithm here
        # change np.inf to 0 in trust_matrix before calling eigentrust
        trust_matrix = np.nan_to_num(trust_matrix, posinf=0.0)
        self.global_values = eigentrust(trust_matrix, pretrusted=self.pretrusted, alpha=self.alpha, eps=1e-10, max_iter=100_000)