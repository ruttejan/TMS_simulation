from dataclasses import dataclass
import random

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
    
    def __init__(self, mu: float, sigma: float):
        self.price_mean = RunningMean()
        self.mu = mu
        self.sigma = sigma

    def update_mean(self, price: float) -> None:
        """Update the running mean price with a new transaction price."""
        self.price_mean.update(price)
        
    def gen_price(self, rng: random.Random) -> float:
        """Generate a new transaction price from a log-normal distribution."""
        return rng.lognormvariate(self.mu, self.sigma)
        
    def weight_from_price(self, price: float) -> float:
        """Compute a weight from the price relative to the running mean price."""    
        if self.price_mean.count > 0:
            p_bar = self.price_mean.mean
        else:
            p_bar = price
        if p_bar <= 0:
            return 1.0
        r = price / p_bar
        return r