from compositeProduct import CompositeProduct

class EquityProduct(CompositeProduct):
    """
    Pour les produits equity composés.
    Garantit que greeks() contient toujours delta, gamma, vega, theta, rho.
    """

    _EQUITY_GREEKS = ("delta", "gamma", "vega", "theta", "rho")

    def greeks(self, **kwargs) -> Dict[str, float]:
        g = super().greeks(**kwargs)
        for key in self._EQUITY_GREEKS:
            g.setdefault(key, 0.0)
        return g

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass