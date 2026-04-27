from abc import ABC, abstractmethod
import numpy as np

# Stochastic Models
class StochasticRateModel(ABC):
    """
    Abstract class for stochastic rate curve models

    - calibrate(): estimation des paramètres sur les données historiques
    - zero_bond_price(): calcul du prix d'un ZC
    - simulate_paths(): simulations de Monte Carlo de chemins de taux
        - return: array(n_simulations, n_steps+1)
    """

    @abstractmethod
    def simulate_paths(self, r0: float, T: float, n_steps: int,
                       n_simulations: int, seed: int = None) -> np.ndarray:
        pass

    @abstractmethod
    def zero_bond_price(self, r0: float, T: float) -> float:
        pass