"""The algo-wheel's pre-trade cost model, imported as fitted there and applied here to net the signal down.

    cost_bp(order) = a + b * sigma_bp * (Q / ADV) ** beta + c * spread_bp

The two parameter sets come from ProjectF's results/models.json and results/pretrade.json: `routed` is the optimiser's
reduced form fitted on orders sent to the wheel's best broker (a = 0.863, b = 0.201, c = 0), `observed` the pre-trade
fit on the desk's observed cost against arrival over 2010-2014 (a = -1.14, b = 0.257, c = 0.79).  Both are one-way
costs of a single order.  For futures the fixed part is half a tick plus a commission per contract, and the impact term
keeps the same b against the contract's own ADV: an assumption, stated in the README."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PARAMS = {
    "routed": {"a": 0.8631, "b": 0.2007, "beta": 0.5, "c": 0.0},
    "observed": {"a": -1.14, "b": 0.257, "beta": 0.5, "c": 0.79},
}
FUTURES_COMMISSION_USD = 2.0       # per contract per side, all-in


@dataclass
class CostModel:
    a: float = 0.8631
    b: float = 0.2007
    beta: float = 0.5
    c: float = 0.0
    name: str = "routed"
    futures: bool = False

    @classmethod
    def named(cls, name: str, universe: str = "equity") -> "CostModel":
        p = PARAMS[name]
        if universe == "futures":                  # Yahoo's contract volumes are not reliable enough for an impact term
            return cls(0.0, 0.0, p["beta"], 0.0, name, futures=True)
        return cls(p["a"], p["b"], p["beta"], p["c"], name, futures=False)

    def cost_bp(self, sigma_bp, spread_bp, pct_adv, notional=None):
        """one-way cost in bp of an order that is pct_adv of the name's ADV; for futures spread_bp is one tick in bp
        and notional the dollar value of a contract (commission is converted to bp of it)"""
        x = np.power(np.maximum(np.asarray(pct_adv, dtype=float), 0.0), self.beta)
        impact = self.b * np.asarray(sigma_bp, dtype=float) * x
        if self.futures:
            fixed = 0.5 * np.asarray(spread_bp, dtype=float) + 1e4 * FUTURES_COMMISSION_USD / np.maximum(np.asarray(notional, dtype=float), 1.0)
            return fixed + impact
        return self.a + impact + self.c * np.asarray(spread_bp, dtype=float)
