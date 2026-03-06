from dataclasses import dataclass
import math

@dataclass
class RunningMean:
    """Numerically stable running mean (online update)."""

    count: int = 0
    mean: float = 0.0

    def update(self, x: float) -> None:
        """Update the running mean with one new observation."""
        self.count += 1
        if self.count == 1:
            self.mean = x
        else:
            self.mean += (x - self.mean) / self.count
            
            
class PriceHandler:
    """Handles price-related computations, such as maintaining a running mean price and computing price weights."""
    
    def __init__(self, r_max: float = 10.0):
        self.price_mean = RunningMean()
        self.r_max = r_max
        
    def update_mean(self, price: float) -> None:
        """Update the running mean price with a new transaction price."""
        self.price_mean.update(price)
        
    def weight_from_price(self, price: float) -> float:
        """Compute bounded price weight $w_k\\in(0,1]$.

        Implements:
            R_k = p_k / p̄
            w_k = log(1 + min(R_k, R_max)) / log(1 + R_max)

        Args:
            price: Transaction price p_k.                
        Returns:
            Price weight w_k in (0, 1].
        """        
        if self.price_mean.count > 0:
            p_bar = self.price_mean.mean
        else:
            p_bar = price
        if p_bar <= 0:
            return 1.0
        r = price / p_bar
        if r < 0:
            r = 0.0
        r = min(r, self.r_max)
        denom = math.log(1.0 + self.r_max)
        if denom <= 0:
            return 1.0
        return math.log(1.0 + r) / denom