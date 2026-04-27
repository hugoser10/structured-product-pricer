import numpy as np
from abc import ABC, abstractmethod


class StochasticRateModel(ABC):
    """Interface commune des modèles stochastiques de taux courts."""

    @abstractmethod
    def simulate_paths(self, r0: float, T: float, n_steps: int,
                       n_simulations: int, seed: int = None) -> np.ndarray:
        ...

    @abstractmethod
    def zero_bond_price(self, r0: float, T: float) -> float:
        ...
