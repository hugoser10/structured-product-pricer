from typing import Dict, Any
from pricer.products.base.rate_product import RateProduct
from pricer.products.rates.coupon_bond import CouponBond
from pricer.products.rates.bond_option import BondOption


class CallableBond(RateProduct):
    """Callable = CouponBond − BondOption(call). L'émetteur peut racheter à `call_price`."""

    def __init__(self, nominal: float, coupon_rate: float, maturity: float,
                 call_dates: list, call_price: float, frequency: int, hw_model):
        super().__init__()
        self.nominal = nominal
        self.coupon_rate = coupon_rate
        self.maturity = maturity
        self._add_leg(CouponBond(nominal, coupon_rate, maturity, frequency,
                                 hw_model.rate_curve), +1.0)
        self._add_leg(BondOption(nominal, coupon_rate, maturity, call_dates,
                                  call_price, frequency, hw_model, "call"), -1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "CallableBond", "nominal": self.nominal,
                "coupon_rate": self.coupon_rate, "maturity": self.maturity}
