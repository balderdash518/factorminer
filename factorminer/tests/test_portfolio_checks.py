"""WorldQuant BRAIN-style book checks (evaluation.portfolio_checks)."""

from __future__ import annotations

import numpy as np
import pytest

from factorminer.agent.prompt_builder import PromptBuilder
from factorminer.application.validation_pipeline import ValidationPipeline
from factorminer.core.types import DEFAULT_FEATURES
from factorminer.evaluation.portfolio_checks import (
    PortfolioThresholds,
    book_stats,
    book_weights,
    liquid_subset,
    run_portfolio_checks,
)
from factorminer.memory.feedback import render_batch_feedback, summarize_trajectory
from factorminer.utils.config import load_config

M, T = 60, 400
CHECKS = PortfolioThresholds()


def _persistent(rng: np.random.Generator, rho: float = 0.99) -> np.ndarray:
    x = np.zeros((M, T))
    x[:, 0] = rng.normal(size=M)
    for t in range(1, T):
        x[:, t] = rho * x[:, t - 1] + np.sqrt(1 - rho**2) * rng.normal(size=M)
    return x


def test_thresholds_are_off_unless_enabled_and_reject_unknown_keys() -> None:
    assert PortfolioThresholds.from_mapping({}) is None
    assert PortfolioThresholds.from_mapping({"enabled": False, "sharpe_min": 2}) is None
    custom = PortfolioThresholds.from_mapping({"enabled": True, "sharpe_min": 2, "liquidity_window": 10})
    assert custom.sharpe_min == 2.0 and custom.liquidity_window == 10
    with pytest.raises(ValueError, match="sharp_min"):
        PortfolioThresholds.from_mapping({"enabled": True, "sharp_min": 2})
    with pytest.raises(ValueError, match="sub_universe_fraction"):
        load_config(overrides={"evaluation": {"portfolio_checks": {
            "enabled": True, "sub_universe_fraction": 0}}}).validate()


def test_book_is_market_neutral_with_unit_gross_and_skips_thin_periods() -> None:
    rng = np.random.default_rng(0)
    signals = rng.normal(size=(M, 5))
    returns = rng.normal(size=(M, 5))
    returns[5:, 4] = np.nan  # only 5 assets: not traded
    w = book_weights(signals, returns)
    np.testing.assert_allclose(w[:, :4].sum(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.abs(w[:, :4]).sum(axis=0), 1.0)
    assert not w[:, 4].any()


def test_book_stats_follow_brain_definitions() -> None:
    rng = np.random.default_rng(1)
    x = _persistent(rng)
    returns = 0.002 * x + 0.02 * rng.normal(size=(M, T))
    stats = book_stats(x, returns)
    w = book_weights(x, returns)
    pnl = (w * returns).sum(axis=0)
    assert stats["sharpe"] == pytest.approx(pnl.mean() / pnl.std(ddof=1) * np.sqrt(252))
    assert stats["returns"] == pytest.approx(pnl.mean() * 252 / 0.5)
    assert stats["turnover"] == pytest.approx(np.abs(np.diff(w, axis=1)).sum(axis=0).mean())
    assert stats["fitness"] == pytest.approx(
        stats["sharpe"] * np.sqrt(abs(stats["returns"]) / max(stats["turnover"], 0.125))
    )
    assert stats["sharpe"] > 3 and stats["turnover"] < 0.3


def test_checks_flip_to_the_ic_side_and_name_each_failure() -> None:
    rng = np.random.default_rng(2)
    x = _persistent(rng)
    returns = 0.002 * x + 0.02 * rng.normal(size=(M, T))
    metrics, failures = run_portfolio_checks(-x, returns, CHECKS, direction=-1.0)
    assert failures == [] and metrics["sharpe"] > 3 and metrics["direction"] == -1.0

    iid = rng.normal(size=(M, T))  # no edge and a fresh book every day
    metrics, failures = run_portfolio_checks(iid, returns, CHECKS)
    assert any(f.startswith("Sharpe") for f in failures)
    assert any(f.startswith("Turnover") and ">" in f for f in failures)

    concentrated = np.zeros((M, T))
    concentrated[0] = 1.0  # all weight on one name after demeaning
    _, failures = run_portfolio_checks(concentrated + 1e-9 * iid, returns, CHECKS)
    assert any(f.startswith("Max weight") for f in failures)


def test_sub_universe_uses_trailing_liquidity_and_brain_cutoff() -> None:
    liquidity = np.tile(np.arange(M, dtype=float)[:, None], (1, T))
    liquidity[:, 200:] = liquidity[::-1, 200:]  # ranking flips at t=200
    mask = liquid_subset(liquidity, 1 / 3, window=20)
    assert mask[:, 50].sum() == M // 3 and mask[-1, 50] and not mask[0, 50]
    assert mask[-1, 205]  # trailing window still mostly sees the old ranking
    assert mask[0, 260] and not mask[-1, 260]

    rng = np.random.default_rng(3)
    x = _persistent(rng)
    returns = 0.002 * x + 0.02 * rng.normal(size=(M, T))
    returns[mask] = 0.02 * rng.normal(size=int(mask.sum()))  # no edge among liquid names
    metrics, failures = run_portfolio_checks(x, returns, CHECKS, liquid_mask=mask)
    assert metrics["sub_universe_cutoff"] == pytest.approx(0.75 * np.sqrt(1 / 3) * metrics["sharpe"])
    assert any(f.startswith("Sub-universe Sharpe") for f in failures)


def _pipeline(checks: PortfolioThresholds | None) -> tuple[ValidationPipeline, np.ndarray]:
    rng = np.random.default_rng(4)
    tensor = np.abs(rng.normal(size=(M, T, len(DEFAULT_FEATURES)))) + 1.0
    signal = _persistent(rng)
    tensor[:, :, DEFAULT_FEATURES.index("$open")] = signal
    returns = 0.002 * signal + 0.02 * rng.normal(size=(M, T))
    pipeline = ValidationPipeline(
        tensor, returns, ic_threshold=0.001, fast_screen_assets=M, portfolio_checks=checks,
    )
    return pipeline, returns


def test_pipeline_rejects_on_failed_checks_and_keeps_metrics() -> None:
    pipeline, _ = _pipeline(CHECKS)
    good, noise = pipeline.evaluate_batch([("edge", "Neg($open)"), ("noise", "Neg($high)")])
    assert good.admitted and good.portfolio["sharpe"] > 3 and good.portfolio["direction"] == -1.0
    assert noise.outcome == "failed_checks"
    assert "Turnover" in noise.rejection_reason  # an iid signal rebuilds the book daily
    assert noise.portfolio is not None

    plain, _ = _pipeline(None)
    result = plain.evaluate_batch([("noise", "Neg($high)")])[0]
    assert result.portfolio is None and result.outcome != "failed_checks"


def test_feedback_and_prompt_show_book_metrics() -> None:
    trajectory = [
        {"factor_id": "a", "formula": "Neg($high)", "ic_paper_mean": 0.02, "outcome": "failed_checks",
         "rejection_reason": "Sharpe 0.80 < 1.25; Fitness 0.40 < 1",
         "portfolio": {"sharpe": 0.8, "fitness": 0.4, "turnover": 0.3}},
        {"factor_id": "b", "formula": "Neg($low)", "ic_paper_mean": 0.03, "outcome": "failed_checks",
         "rejection_reason": "Sharpe 0.50 < 1.25; Turnover 85.0% > 70%",
         "portfolio": {"sharpe": 0.5, "fitness": 0.2, "turnover": 0.85}},
    ]
    feedback = summarize_trajectory(trajectory, 1)
    text = render_batch_feedback(feedback)
    assert "2 candidates: failed_checks=2" in text
    assert "failing checks: Sharpe 2, Fitness 1, Turnover 1" in text
    assert "Closest to the portfolio checks but rejected:" in text
    # Ranked by Sharpe, the binding gate, not by |IC|.
    assert text.index("Neg($high)") < text.index("Neg($low)")
    assert "Sharpe 0.80, Fitness 0.40, Turnover 30%" in text

    prompt = PromptBuilder().build_user_prompt(
        {}, {"library_size": 0, "ic_threshold": 0.001, "correlation_threshold": 0.7,
             "portfolio_checks": CHECKS.describe()}, batch_size=4,
    )
    assert "annualized Sharpe >= 1.25, Fitness >= 1" in prompt
    assert "daily turnover between 1% and 70%" in prompt
