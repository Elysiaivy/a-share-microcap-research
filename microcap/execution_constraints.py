"""Optional order-size/cash-cost sensitivity, retaining total-return accounting.

Adjusted units are synthetic claims: factor changes still reinvest distributions.
This layer constrains orders; it does not reproduce a broker corporate-action ledger.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class OrderConstraints:
    capital: float = 1_000_000
    lot: int = 100
    minimum_commission: float = 0
    commission_bps: float = 2.5

    def validate(self, rate):
        if self.capital <= 0 or self.lot < 0 or self.minimum_commission < 0:
            raise ValueError('Invalid execution constraint')
        if not 0 <= self.commission_bps / 10000 <= rate:
            raise ValueError('Commission component must be within all-in variable cost')

    def fee(self, amount, rate):
        amount = np.asarray(amount)
        extra = np.maximum(self.minimum_commission - amount * self.commission_bps / 10000, 0)
        return np.where(amount > 0, amount * rate + extra, 0)

    def round_units(self, quantity, factor):
        if not self.lot:
            return quantity
        return np.floor(quantity * factor / self.lot + 1e-10) * self.lot / factor

    def affordable(self, quantity, factor, limit, budget, rate):
        # Both inequalities must hold because the fee is a maximum of two lines.
        notional = min(budget / (1 + rate),
                       max(0, budget - self.minimum_commission) /
                       (1 + rate - self.commission_bps / 10000))
        return self.round_units(min(quantity, notional / limit), factor)
