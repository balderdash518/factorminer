"""``data.features`` decides which leaves the prompt offers and how tensor slices are named."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factorminer.agent.factor_generator import FactorGenerator, restrict_features
from factorminer.agent.output_parser import parse_llm_output
from factorminer.agent.prompt_builder import PromptBuilder, build_system_prompt
from factorminer.core.ralph_loop import RalphLoop
from factorminer.core.types import DEFAULT_FEATURES, configured_feature_names, reset_features
from factorminer.evaluation.runtime import (
    _column_to_feature,
    _resolve_feature_columns,
    load_runtime_dataset,
)
from factorminer.utils.config import load_config

NO_VWAP = [f for f in DEFAULT_FEATURES if f != "$vwap"]


@pytest.fixture(autouse=True)
def _reset_feature_registry():
    reset_features()
    yield
    reset_features()


def test_configured_names_follow_runtime_column_order() -> None:
    features = ["$close", "amount", "$amt", "$volume", "$returns"]
    assert configured_feature_names(features) == [
        _column_to_feature(c) for c in _resolve_feature_columns(features)
    ] == ["$close", "$amt", "$volume", "$returns"]
    assert configured_feature_names([]) == list(DEFAULT_FEATURES)


def test_system_prompt_offers_only_configured_features() -> None:
    # The default prompt must stay byte-identical (managed prompt hash).
    assert build_system_prompt(features=list(DEFAULT_FEATURES)) == build_system_prompt()
    prompt = build_system_prompt(features=NO_VWAP)
    assert "$vwap" not in prompt
    assert "Neg(CsRank(Delta($close, 5)))" in prompt
    assert "($amt, and any registered alt-data leaves are underused)" in prompt


def test_unavailable_leaves_are_sent_to_repair_with_the_reason() -> None:
    candidates, _ = parse_llm_output(
        "1. dev: CsRank(Sub($close, $vwap))\n2. rev: Neg(CsRank($returns))"
    )
    restrict_features(candidates, NO_VWAP)
    assert [c.is_valid for c in candidates] == [False, True]
    assert candidates[0].parse_error.startswith("$vwap not available in this dataset; use only $open")
    candidates, _ = parse_llm_output("1. dev: CsRank(Sub($close, $vwap))")
    assert restrict_features(candidates, None)[0].is_valid


class _ScriptedProvider:
    provider_name = "scripted"
    model = "scripted"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.last_call: dict = {}

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        self.prompts.append(user_prompt)
        return self.replies.pop(0)


def test_generator_repairs_a_formula_that_uses_a_dropped_feature() -> None:
    provider = _ScriptedProvider([
        "1. dev: CsRank(Sub($close, $vwap))\n2. rev: Neg(CsRank($returns))",
        "1. dev: CsRank(Sub($close, Mean($close, 5)))",
    ])
    generator = FactorGenerator(provider, PromptBuilder(features=NO_VWAP))
    batch = generator.generate_batch({}, {}, batch_size=2)
    assert sorted(c.formula for c in batch) == [
        "CsRank(Sub($close, Mean($close, 5)))", "Neg(CsRank($returns))",
    ]
    assert "$vwap not available in this dataset" in provider.prompts[1]


def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for asset in ("A", "B", "C", "D", "E"):
        close = 10.0 + np.cumsum(rng.normal(scale=0.1, size=60))
        for i, dt in enumerate(pd.bdate_range("2024-01-02", periods=60)):
            rows.append({
                "datetime": dt, "asset_id": asset, "open": close[i] - 0.05,
                "high": close[i] + 0.2, "low": close[i] - 0.2, "close": close[i],
                "volume": 1000.0 + i, "amount": close[i] * (1000.0 + i),
            })
    return pd.DataFrame(rows)


def test_loop_names_tensor_slices_from_data_features(tmp_path) -> None:
    cfg = load_config(overrides={"data": {
        "features": NO_VWAP,
        "train_period": ["2024-01-02", "2024-02-29"],
        "test_period": ["2024-03-01", "2024-03-29"],
    }})
    cfg.output_dir = str(tmp_path)
    dataset = load_runtime_dataset(_panel(), cfg)
    assert dataset.data_tensor.shape[2] == 7
    loop = RalphLoop(cfg, dataset.data_tensor, dataset.returns)

    assert loop.feature_names == NO_VWAP
    assert "$vwap" not in loop.generator.prompt_builder.system_prompt
    data = loop.pipeline._build_data_dict()
    assert set(data) == set(NO_VWAP)
    # Before, slice 6 was read as $vwap and $returns was missing.
    np.testing.assert_array_equal(data["$returns"], dataset.data_dict["$returns"])


def test_loop_keeps_registry_order_when_tensor_width_differs(tmp_path) -> None:
    cfg = load_config(overrides={"data": {"features": NO_VWAP}})
    cfg.output_dir = str(tmp_path)
    rng = np.random.default_rng(0)
    loop = RalphLoop(cfg, rng.normal(size=(5, 20, 8)), rng.normal(size=(5, 20)))
    assert loop.feature_names == list(DEFAULT_FEATURES)
