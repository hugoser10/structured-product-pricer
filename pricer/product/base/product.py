from abc import ABC, abstractmethod
from typing import Dict, Any

class Product(ABC):
    """Interface : price() / greeks() / to_dict()."""

    @abstractmethod
    def price(self, **kwargs) -> float: ...

    @abstractmethod
    def greeks(self, **kwargs) -> Dict[str, float]: ...

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]: ...

    def __repr__(self):
        return f"{self.__class__.__name__}({self.to_dict()})"
