"""WorldQuant BRAIN-style portfolio checks for a candidate factor.

The paper's admission gate is |IC| plus library correlation. BRAIN instead
simulates the alpha as a book and gates on the book's statistics. This module
reproduces those checks on the mining panel:

* the signal is demeaned across assets each period (market neutral) and scaled
  so the gross book ``sum |w|`` is 1;
* ``pnl[t] = sum_i w[i, t] * r[i, t]`` with ``r`` the evaluation target, so the
  delay/holding convention is the same one the IC uses;
* Sharpe = sqrt(P) * mean(pnl) / std(pnl), P periods per year;
* Returns = P * mean(pnl) / 0.5 (BRAIN reports PnL on half the book);
* Turnover = mean daily ``sum |w[t] - w[t-1]|`` (traded value / book);
* Fitness = Sharpe * sqrt(|Returns| / max(Turnover, 0.125));
* sub-universe Sharpe uses only the most liquid ``sub_universe_fraction`` of
  assets each period and must reach
  ``sub_universe_sharpe_ratio * sqrt(fraction) * Sharpe``;
* weight concentration is the mean over periods of the largest |w|.

Mining scores |IC|, so the book takes the side the IC points to (``direction``);
a factor whose negation passes is reported with direction -1. Self-correlation
stays with the library's correlation gate (signal Spearman, not PnL).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np

_MIN_ASSETS = 10  # periods with fewer valid assets are not traded


@dataclass(frozen=True)
class PortfolioThresholds:
    """Cutoffs; defaults follow the BRAIN submission checks."""

    sharpe_min: float = 1.25
    fitness_min: float = 1.0
    turnover_min: float = 0.01
    turnover_max: float = 0.70
    max_weight: float = 0.10
    sub_universe_fraction: float = 1.0 / 3.0
    sub_universe_sharpe_ratio: float = 0.75
    liquidity_window: int = 20
    periods_per_year: float = 252.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> PortfolioThresholds | None:
        """``None`` unless ``raw`` has ``enabled: true``; unknown keys raise."""
        raw = dict(raw or {})
        if not raw.pop("enabled", False):
            return None
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"unknown evaluation.portfolio_checks keys: {', '.join(unknown)}")
        thresholds = cls(**{
            key: int(value) if key == "liquidity_window" else float(value)
            for key, value in raw.items()
        })
        if not 0.0 < thresholds.sub_universe_fraction <= 1.0:
            raise ValueError("portfolio_checks.sub_universe_fraction must be in (0, 1]")
        if not 0.0 <= thresholds.turnover_min < thresholds.turnover_max:
            raise ValueError("portfolio_checks needs 0 <= turnover_min < turnover_max")
        return thresholds

    def describe(self) -> str:
        """One sentence for the generation prompt."""
        return (
            f"It must also pass portfolio checks on a market-neutral book: "
            f"annualized Sharpe >= {self.sharpe_min:g}, Fitness >= {self.fitness_min:g} "
            f"(Sharpe * sqrt(|Returns| / max(Turnover, 0.125))), daily turnover between "
            f"{self.turnover_min:.0%} and {self.turnover_max:.0%}, Sharpe on the most liquid "
            f"{self.sub_universe_fraction:.0%} of stocks >= "
            f"{self.sub_universe_sharpe_ratio * np.sqrt(self.sub_universe_fraction):.2f} x Sharpe, "
            f"and no stock above {self.max_weight:.0%} of the book on average."
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def book_weights(signals: np.ndarray, returns: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Market-neutral weights with unit gross book per period (0 where not traded)."""
    valid = np.isfinite(signals) & np.isfinite(returns)
    if mask is not None:
        valid &= mask
    with np.errstate(invalid="ignore"):
        x = np.where(valid, signals, np.nan)
        x = x - np.nanmean(x, axis=0, keepdims=True)
        gross = np.nansum(np.abs(x), axis=0, keepdims=True)
        traded = (valid.sum(axis=0, keepdims=True) >= _MIN_ASSETS) & (gross > 1e-12)
        w = np.where(traded, x / np.where(gross > 1e-12, gross, 1.0), np.nan)
    return np.nan_to_num(w, nan=0.0)


def book_stats(
    signals: np.ndarray,
    returns: np.ndarray,
    *,
    periods_per_year: float = 252.0,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Sharpe, Returns, Turnover, Fitness and weight concentration of the book."""
    w = book_weights(signals, returns, mask)
    traded = np.abs(w).sum(axis=0) > 0
    if traded.sum() < 2:
        return {"sharpe": 0.0, "returns": 0.0, "turnover": 0.0, "fitness": 0.0,
                "max_weight": 0.0, "periods": int(traded.sum())}
    pnl = (w * np.nan_to_num(returns, nan=0.0)).sum(axis=0)[traded]
    std = float(np.std(pnl, ddof=1))
    sharpe = float(np.mean(pnl) / std * np.sqrt(periods_per_year)) if std > 1e-12 else 0.0
    annual_returns = float(np.mean(pnl) * periods_per_year / 0.5)
    trades = np.abs(np.diff(w, axis=1)).sum(axis=0)
    turnover = float(np.mean(trades[traded[1:]])) if traded[1:].any() else 0.0
    fitness = sharpe * float(np.sqrt(abs(annual_returns) / max(turnover, 0.125)))
    max_weight = float(np.mean(np.abs(w).max(axis=0)[traded]))
    return {"sharpe": sharpe, "returns": annual_returns, "turnover": turnover,
            "fitness": fitness, "max_weight": max_weight, "periods": int(traded.sum())}


def liquid_subset(liquidity: np.ndarray, fraction: float, window: int) -> np.ndarray:
    """Mask of the most liquid ``fraction`` of assets each period.

    Liquidity is the trailing ``window``-period mean of ``liquidity`` (e.g.
    close * volume), so the subset never uses future data.
    """
    m, t = liquidity.shape
    values = np.where(np.isfinite(liquidity), liquidity, 0.0)
    counts = np.isfinite(liquidity).astype(float)
    cum_v = np.concatenate([np.zeros((m, 1)), np.cumsum(values, axis=1)], axis=1)
    cum_c = np.concatenate([np.zeros((m, 1)), np.cumsum(counts, axis=1)], axis=1)
    starts = np.maximum(np.arange(1, t + 1) - window, 0)
    ends = np.arange(1, t + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        trailing = (cum_v[:, ends] - cum_v[:, starts]) / (cum_c[:, ends] - cum_c[:, starts])
    mask = np.zeros((m, t), dtype=bool)
    for col in range(t):
        finite = np.flatnonzero(np.isfinite(trailing[:, col]) & (trailing[:, col] > 0))
        keep = int(np.ceil(len(finite) * fraction))
        if keep:
            top = finite[np.argsort(trailing[finite, col])[::-1][:keep]]
            mask[top, col] = True
    return mask


def run_portfolio_checks(
    signals: np.ndarray,
    returns: np.ndarray,
    thresholds: PortfolioThresholds,
    *,
    direction: float = 1.0,
    liquid_mask: np.ndarray | None = None,
) -> tuple[dict[str, float], list[str]]:
    """Return (metrics, failed checks); no failures means the factor passes."""
    oriented = signals * (1.0 if direction >= 0 else -1.0)
    metrics = book_stats(oriented, returns, periods_per_year=thresholds.periods_per_year)
    metrics["direction"] = 1.0 if direction >= 0 else -1.0
    failures: list[str] = []
    if metrics["sharpe"] < thresholds.sharpe_min:
        failures.append(f"Sharpe {metrics['sharpe']:.2f} < {thresholds.sharpe_min:g}")
    if metrics["fitness"] < thresholds.fitness_min:
        failures.append(f"Fitness {metrics['fitness']:.2f} < {thresholds.fitness_min:g}")
    if metrics["turnover"] < thresholds.turnover_min:
        failures.append(f"Turnover {metrics['turnover']:.1%} < {thresholds.turnover_min:.0%}")
    if metrics["turnover"] > thresholds.turnover_max:
        failures.append(f"Turnover {metrics['turnover']:.1%} > {thresholds.turnover_max:.0%}")
    if metrics["max_weight"] > thresholds.max_weight:
        failures.append(f"Max weight {metrics['max_weight']:.1%} > {thresholds.max_weight:.0%}")
    if liquid_mask is not None and thresholds.sub_universe_fraction < 1.0:
        sub = book_stats(oriented, returns, periods_per_year=thresholds.periods_per_year,
                         mask=liquid_mask)
        cutoff = (thresholds.sub_universe_sharpe_ratio
                  * np.sqrt(thresholds.sub_universe_fraction) * metrics["sharpe"])
        metrics["sub_universe_sharpe"] = sub["sharpe"]
        metrics["sub_universe_cutoff"] = float(cutoff)
        if sub["sharpe"] < cutoff:
            failures.append(f"Sub-universe Sharpe {sub['sharpe']:.2f} < {cutoff:.2f}")
    return metrics, failures
