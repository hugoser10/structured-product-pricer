from typing import Dict, Any
from pricer.products.base.rate_product import RateProduct
from pricer.products.rates.coupon_bond import CouponBond
from pricer.products.rates.bond_option import BondOption


class PutableBond(RateProduct):
    """Putable = CouponBond + BondOption(put). L'investisseur peut revendre à `put_price`."""

    def __init__(self, nominal: float, coupon_rate: float, maturity: float,
                 put_dates: list, put_price: float, frequency: int, hw_model):
        super().__init__()
        self.nominal = nominal
        self.coupon_rate = coupon_rate
        self.maturity = maturity
        self._add_leg(CouponBond(nominal, coupon_rate, maturity, frequency,
                                 hw_model.rate_curve), +1.0)
        self._add_leg(BondOption(nominal, coupon_rate, maturity, put_dates,
                                  put_price, frequency, hw_model, "put"), +1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "PutableBond", "nominal": self.nominal,
                "coupon_rate": self.coupon_rate, "maturity": self.maturity}
