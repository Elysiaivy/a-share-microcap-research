"""Daily aggregate liquidity pressure scenarios, not a certified fill model.

Orders are planned before the session. Realized amount/volume determine only
VWAP reference fills and a daily aggregate capacity ceiling. The square-root
charge is incremental to the original fee model, uncalibrated and paid once.
"""
from dataclasses import dataclass
import numpy as np


VWAP_ABSOLUTE_TOLERANCE = .011
VWAP_RELATIVE_TOLERANCE = 1e-5


def vwap_price_tolerance(price):
    """Allow only provider rounding tolerance in the daily high/low check."""
    return np.maximum(VWAP_ABSOLUTE_TOLERANCE, np.abs(price) * VWAP_RELATIVE_TOLERANCE)


@dataclass(frozen=True)
class LiquidityConstraints:
    participation: float | None = .01
    impact_eta: float = .5
    require_known_sigma: bool = False

    def validate(self, market):
        if self.participation is not None and not 0 < self.participation <= 1:
            raise ValueError('Participation must be in (0, 1] or None')
        if not np.isfinite(self.impact_eta) or self.impact_eta < 0:
            raise ValueError('Impact coefficient must be finite and nonnegative')
        shape = market['open'].shape
        for name in ['amount_yuan', 'volume_shares', 'high', 'low', 'adv20', 'sigma20']:
            if 'capacity_' + name not in market or market['capacity_' + name].shape != shape:
                raise ValueError('Missing or misaligned liquidity array: capacity_' + name)

    def charge(self, amount, adv, sigma, already=0.):
        """Marginal charge on aggregate absolute notional; splitting cannot save it."""
        if amount <= 0 or self.impact_eta == 0:
            return 0.
        if not np.isfinite(adv) or adv <= 0 or not np.isfinite(sigma) or sigma < 0:
            raise ValueError('Impact requires positive lagged ADV and nonnegative volatility')
        return float(self.impact_eta * sigma / np.sqrt(adv) *
                     ((already + amount) ** 1.5 - already ** 1.5))

    def affordable(self, quantity, factor, limit, budget, base_fee, adv, sigma,
                   implementation=None, already=0.):
        """Pre-session upper-price budget, with no realized daily inputs.

        Missing required lagged inputs leave an auditable planned attempt,
        which the session rejects. The default unconstrained, zero-impact VWAP
        control adds no lagged-liquidity filter. ``require_known_sigma`` enables
        a matched-eligibility zero-impact comparison. Missing inputs are never inferred.
        """
        if quantity <= 0 or budget <= 0 or not np.isfinite(limit) or limit <= 0:
            return 0.
        valid = np.isfinite(adv) and adv > 0 and np.isfinite(sigma) and sigma >= 0
        def spend(q):
            amount = q * limit
            impact = self.charge(amount, adv, sigma, already) if valid else 0.
            return amount + float(base_fee(amount)) + impact
        if implementation is not None and implementation.lot:
            count = int(np.floor(quantity * factor / implementation.lot + 1e-10))
            lo, hi = 0, max(count, 0)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                q = mid * implementation.lot / factor
                if spend(q) <= budget + max(1e-10, budget * 1e-12): lo = mid
                else: hi = mid - 1
            return lo * implementation.lot / factor
        if spend(quantity) <= budget:
            return quantity
        lo, hi = 0., quantity
        for _ in range(55):
            mid = (lo + hi) / 2
            if spend(mid) <= budget: lo = mid
            else: hi = mid
        return lo

    def session(self, market, t):
        return LiquiditySession(self, market, t)


class LiquiditySession:
    def __init__(self, rules, market, t):
        self.rules = rules
        self.adv = market['capacity_adv20'][t]
        self.sigma = market['capacity_sigma20'][t]
        self.amount = market['capacity_amount_yuan'][t]
        self.volume = market['capacity_volume_shares'][t]
        self.high = market['capacity_high'][t]
        self.low = market['capacity_low'][t]
        with np.errstate(divide='ignore', invalid='ignore'):
            self.raw_vwap = self.amount / self.volume
        self.price = self.raw_vwap * market['factor'][t]
        self.used = np.zeros_like(self.amount, dtype=float)
        self.valid_lag = np.ones_like(self.amount, dtype=bool)
        if rules.participation is not None or rules.impact_eta > 0:
            self.valid_lag &= np.isfinite(self.adv) & (self.adv > 0)
        if rules.impact_eta > 0 or rules.require_known_sigma:
            self.valid_lag &= np.isfinite(self.sigma) & (self.sigma >= 0)
        tolerance = vwap_price_tolerance(self.raw_vwap)
        self.valid_day = (np.isfinite(self.amount) & (self.amount > 0) &
                          np.isfinite(self.volume) & (self.volume > 0) &
                          np.isfinite(self.high) & np.isfinite(self.low) &
                          (self.low > 0) & (self.high >= self.low) &
                          np.isfinite(self.price) & (self.price > 0) &
                          (self.raw_vwap >= self.low - tolerance) &
                          (self.raw_vwap <= self.high + tolerance))
        self.suspended = market.get('suspended_any', market['suspended'])[t]
        self.up, self.down = market['up'][t], market['down'][t]
        self.capacity = (np.full_like(self.amount, np.inf) if rules.participation is None else
                         rules.participation * np.minimum(self.adv, self.amount))
        self.capacity[~self.valid_lag | ~self.valid_day] = 0.

    def rejection(self, j, side):
        if not self.valid_lag[j]: return 'missing_or_zero_lag_liquidity'
        if not self.valid_day[j]: return 'missing_or_invalid_daily_liquidity'
        if self.suspended[j]: return 'intraday_suspension'
        if side == 'buy' and self.low[j] >= self.up[j] - 1e-7: return 'one_price_upper_limit'
        if side == 'sell' and self.high[j] <= self.down[j] + 1e-7: return 'one_price_lower_limit'
        return None

    def fill_quantity(self, j, planned, factor, implementation=None, full_exit=False):
        remaining = max(0., self.capacity[j] - self.used[j])
        quantity = min(planned, remaining / self.price[j])
        # Odd residuals can be cleared only in full; a capacity-truncated sell
        # is a partial trade and must also meet the same lot convention.
        if implementation is not None and implementation.lot and (not full_exit or quantity < planned - 1e-12):
            quantity = implementation.round_units(quantity, factor)
        return max(0., quantity)

    def impact(self, j, amount):
        return self.rules.charge(amount, self.adv[j], self.sigma[j], self.used[j])

    def consume(self, j, amount):
        self.used[j] += amount
        if self.used[j] > self.capacity[j] + max(1e-8, self.used[j] * 1e-10):
            raise ValueError('Daily aggregate participation ceiling exceeded')

    def audit(self, j, planned, filled, previous_close, **fields):
        return dict(planned_units=planned, filled_units=filled, previous_close=float(previous_close),
                    requested_reference_notional=planned * previous_close,
                    filled_reference_notional=filled * previous_close,
                    adv20=float(self.adv[j]), sigma20=float(self.sigma[j]),
                    reference_vwap=float(self.raw_vwap[j]), day_amount=float(self.amount[j]),
                    capacity_total=float(self.capacity[j]), **fields)
