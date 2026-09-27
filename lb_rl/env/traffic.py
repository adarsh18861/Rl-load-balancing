"""Arrival processes. Each slot produces at most one arrival (Bernoulli),
so the steady-traffic case has an exact transition model (see models/).

All processes expose:
    - rate(slot) -> the arrival probability that would be used at `slot`
    - sample(rng, slot) -> bool, whether a job arrives this slot
Bursty and cyclical processes are stateful only in `sample` (bursty keeps a
Markov regime); `rate` reports the *current* regime's rate for bursty.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SteadyTraffic:
    """Bernoulli arrivals at a fixed rate."""

    lam: float

    def rate(self, slot: int) -> float:
        return self.lam

    def sample(self, rng: np.random.Generator, slot: int) -> bool:
        return bool(rng.random() < self.lam)

    def reset(self) -> None:
        pass


@dataclass
class BurstyTraffic:
    """2-state Markov-modulated Bernoulli arrivals (low/high rate regime)."""

    lam_low: float
    lam_high: float
    p_low_to_high: float
    p_high_to_low: float
    _high: bool = field(default=False, init=False, repr=False)

    def reset(self) -> None:
        self._high = False

    def rate(self, slot: int) -> float:
        return self.lam_high if self._high else self.lam_low

    def sample(self, rng: np.random.Generator, slot: int) -> bool:
        p_switch = self.p_high_to_low if self._high else self.p_low_to_high
        if rng.random() < p_switch:
            self._high = not self._high
        return bool(rng.random() < self.rate(slot))


@dataclass
class CyclicalTraffic:
    """Arrival rate follows a sinusoid: lam(t) = mid + amp * sin(2*pi*t/period)."""

    lam_min: float
    lam_max: float
    period: int

    def rate(self, slot: int) -> float:
        mid = (self.lam_max + self.lam_min) / 2.0
        amp = (self.lam_max - self.lam_min) / 2.0
        return mid + amp * np.sin(2 * np.pi * slot / self.period)

    def sample(self, rng: np.random.Generator, slot: int) -> bool:
        return bool(rng.random() < self.rate(slot))

    def reset(self) -> None:
        pass


def make_traffic(kind: str, **kwargs):
    """Factory from a config dict's `traffic.kind` + kwargs."""
    if kind == "steady":
        return SteadyTraffic(lam=kwargs["lam"])
    if kind == "bursty":
        return BurstyTraffic(
            lam_low=kwargs["lam_low"],
            lam_high=kwargs["lam_high"],
            p_low_to_high=kwargs["p_low_to_high"],
            p_high_to_low=kwargs["p_high_to_low"],
        )
    if kind == "cyclical":
        return CyclicalTraffic(
            lam_min=kwargs["lam_min"], lam_max=kwargs["lam_max"], period=kwargs["period"]
        )
    raise ValueError(f"unknown traffic kind: {kind}")
