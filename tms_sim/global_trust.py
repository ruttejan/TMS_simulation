import numpy as np

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
        self.global_values = np.mean(trust_matrix, axis=1)
        
    def resize_vector(self, new_size: int) -> None:
        """Resize the global trust vector to accommodate more sellers."""
        if new_size > self.n:
            new_global_values = np.zeros(new_size)
            new_global_values[:self.n] = self.global_values
            self.global_values = new_global_values
            self.n = new_size
            
class SHAPETrustStore (GlobalTrustStore):
    """Maintains a vector of global trust values for all sellers using SHAPE method."""
    
    def update(self, trust_matrix: np.ndarray) -> None:
        """Update global trust values based on the trust matrix using SHAPE method."""
        pass # Implement calling of the SHAPE-Trust algorithm here
    
class EigenTrustStore (GlobalTrustStore):
    """Maintains a vector of global trust values for all sellers using EigenTrust method."""
    
    def update(self, trust_matrix: np.ndarray) -> None:
        """Update global trust values based on the trust matrix using EigenTrust method."""
        pass # Implement calling of the EigenTrust algorithm here