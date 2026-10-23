"""Paper-fidelity fixes and the structured feedback path tau -> memory -> prompt."""

from __future__ import annotations

import numpy as np
import pytest

from factorminer.agent.prompt_builder import PromptBuilder
from factorminer.agent.reflection import parse_reflection
from factorminer.application.mining_budget import EvaluationResult
from factorminer.application.validation_pipeline import ValidationPipeline
from factorminer.core.factor_library import Factor, FactorLibrary
from factorminer.core.ralph_loop import RalphLoop
from factorminer.domain.dependence import (
    MeanAbsSpearmanDependenceMetric,
    SpearmanDependenceMetric,
    build_dependence_metric,
)
from factorminer.evaluation.dependence_index import indexed_dependence_metric
from factorminer.memory.evolution import evolve_memory
from factorminer.memory.feedback import render_batch_feedback, summarize_trajectory
from factorminer.memory.formation import form_memory
from factorminer.memory.memory_store import ExperienceMemory
from factorminer.memory.retrieval import retrieve_memory
from factorminer.utils.config import load_config


def _sign_flipping_pair(m: int = 20, t: int = 40) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    a = rng.normal(size=(m, t))
    b = a.copy()
    b[:, 1::2] *= -1.0  # perfectly correlated on even periods, anti on odd
    return a, b


# ---------------------------------------------------------------------------
# Paper eq. (3): rho = mean_t Corr_t, redundancy uses |rho|
# ---------------------------------------------------------------------------


def test_spearman_averages_signed_correlations_like_paper_eq3() -> None:
    a, b = _sign_flipping_pair()
    assert SpearmanDependenceMetric().compute(a, b) == pytest.approx(0.0, abs=1e-12)
    assert MeanAbsSpearmanDependenceMetric().compute(a, b) == pytest.approx(1.0)
    # A consistently negative relation is still fully redundant.
    assert SpearmanDependenceMetric().compute(a, -a) == pytest.approx(1.0)


@pytest.mark.parametrize("name", ["spearman", "mean_abs_spearman"])
def test_indexed_metric_matches_reference_for_both_aggregations(name: str) -> None:
    rng = np.random.default_rng(11)
    a = rng.normal(size=(15, 300))
    b = 0.3 * a + rng.normal(size=(15, 300))
    b[:, ::3] *= -1
    a[rng.random(a.shape) < 0.1] = np.nan
    reference = build_dependence_metric(name)
    indexed = indexed_dependence_metric(name)
    assert indexed.name == reference.name
    assert indexed.compute(a, b) == pytest.approx(reference.compute(a, b), abs=1e-12)
    # Cached second call returns the same aggregation.
    assert indexed.compute(b, a) == pytest.approx(reference.compute(a, b), abs=1e-12)


# ---------------------------------------------------------------------------
# Validation outcomes and trajectory tau
# ---------------------------------------------------------------------------


def _library_with(signals: np.ndarray, *, ic: float = 0.05) -> FactorLibrary:
    library = FactorLibrary(correlation_threshold=0.5, ic_threshold=0.01)
    library.admit_factor(
        Factor(
            id=0, name="seed", formula="Neg($close)", category="Other",
            ic_mean=ic, icir=1.0, ic_win_rate=0.6, max_correlation=0.0,
            batch_number=0, ic_paper_mean=abs(ic), signals=signals,
        )
    )
    return library


def _pipeline(library: FactorLibrary, returns: np.ndarray) -> ValidationPipeline:
    m, t = returns.shape
    data = {"$close": np.cumsum(returns, axis=1) + 10.0}
    return ValidationPipeline(
        data_tensor=data, returns=returns, library=library,
        ic_threshold=0.01, replacement_ic_min=0.5, fast_screen_assets=0,
    )


def test_redundant_candidate_records_conflicting_factor_and_outcome() -> None:
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.02, (30, 120))
    library = _library_with(returns.copy(), ic=0.9)
    pipeline = _pipeline(library, returns)

    result = pipeline.evaluate_candidate("copy", "Neg($close)", precomputed=(returns.copy(), None))

    assert result.outcome == "redundant"
    assert result.correlated_with == "#1 seed"
    assert result.conflict_count == 1
    assert result.max_correlation == pytest.approx(1.0)
    assert "insufficient to replace factor 1" in result.replacement_note


def test_batch_dedup_keeps_stronger_absolute_ic_signal() -> None:
    pipeline = _pipeline(FactorLibrary(correlation_threshold=0.5, ic_threshold=0.01),
                         np.zeros((5, 10)))
    signals = np.random.default_rng(1).normal(size=(5, 10))
    strong_negative = EvaluationResult("neg", "A", admitted=True, ic_mean=-0.10,
                                       ic_paper_mean=0.10, signals=signals, stage_passed=3)
    weak_positive = EvaluationResult("pos", "B", admitted=True, ic_mean=0.05,
                                     ic_paper_mean=0.05, signals=signals.copy(), stage_passed=3)

    results = pipeline._deduplicate_batch([weak_positive, strong_negative])

    assert strong_negative.admitted and not weak_positive.admitted
    assert weak_positive.outcome == "batch_duplicate"
    assert weak_positive.duplicate_of == "neg"
    assert len(results) == 2


def test_trajectory_carries_redundancy_diagnostics(tmp_path) -> None:
    rng = np.random.default_rng(2)
    tensor = rng.normal(size=(10, 30, 8))
    returns = rng.normal(0, 0.02, (10, 30))
    cfg = load_config(overrides={"mining": {"ic_threshold": 0.01}})
    cfg.output_dir = str(tmp_path)
    loop = RalphLoop(cfg, tensor, returns)
    result = EvaluationResult(
        "cand", "Neg($close)", parse_ok=True, ic_paper_mean=0.06, max_correlation=0.83,
        correlated_with="#4 vwap_dev", conflict_count=1, outcome="redundant",
        rejection_reason="Max spearman dependence 0.8300 >= threshold 0.5",
        replacement_note="IC 0.0600 below replacement floor 0.1",
    )

    entry = loop._build_trajectory([result])[0]

    assert entry["max_correlation"] == 0.83
    assert entry["correlated_with"] == "#4 vwap_dev"
    assert entry["outcome"] == "redundant"
    assert entry["replacement_note"].startswith("IC 0.0600")


# ---------------------------------------------------------------------------
# Formation / evolution
# ---------------------------------------------------------------------------


def _trajectory() -> list[dict]:
    return [
        {"factor_id": "a", "formula": "CsRank(Div(Sub($close, $vwap), $vwap))", "admitted": True,
         "paper_ic": 0.07, "paper_icir": 0.6, "max_correlation": 0.2, "outcome": "admitted"},
        {"factor_id": "b", "formula": "Neg(Div(Sub($close, $vwap), $vwap))", "admitted": False,
         "paper_ic": 0.06, "max_correlation": 0.91, "correlated_with": "#1 a",
         "conflict_count": 1, "outcome": "redundant",
         "rejection_reason": "Max spearman dependence 0.9100 >= threshold 0.5"},
        {"factor_id": "c", "formula": "Mean($close, 5)", "admitted": False, "paper_ic": 0.004,
         "outcome": "low_ic", "rejection_reason": "Paper IC 0.0040 < threshold 0.04"},
        {"factor_id": "d", "formula": "Foo($close)", "admitted": False, "outcome": "parse_error",
         "rejection_reason": "Parse failure"},
    ]


def test_dependence_rejections_populate_forbidden_directions() -> None:
    formed = form_memory(ExperienceMemory(), _trajectory(), batch_number=1)
    names = [d.name for d in formed.forbidden_directions]
    assert names == ["VWAP Deviation variants"]
    direction = formed.forbidden_directions[0]
    assert direction.correlated_factors == ["#1 a"]
    assert direction.typical_correlation == pytest.approx(0.91)


def test_low_ic_and_parse_failures_stay_out_of_forbidden_directions() -> None:
    trajectory = [e for e in _trajectory() if e["outcome"] in {"low_ic", "parse_error"}]
    formed = form_memory(ExperienceMemory(), trajectory, batch_number=1)
    assert formed.forbidden_directions == []


def test_repeated_batches_accumulate_counts_without_doubling() -> None:
    memory = ExperienceMemory()
    for batch in range(1, 4):
        memory = evolve_memory(memory, form_memory(memory, _trajectory(), batch_number=batch))
    direction = next(d for d in memory.forbidden_directions if d.name == "VWAP Deviation variants")
    assert direction.occurrence_count == 3
    assert len(memory.state.feedback_history) == 3


def test_keyword_families_require_their_anchor_tokens() -> None:
    from factorminer.memory.formation import (
        _classify_forbidden_direction,
        _classify_success_pattern,
    )

    assert _classify_forbidden_direction("Neg(CsRank(Delta($close, 5)))") == "Simple Delta Reversal"
    assert _classify_forbidden_direction("CsRank(Sub($close, Delta($close, 3)))") is None
    # "Or" must not match inside "Corr".
    assert _classify_success_pattern("Neg(Corr($close, $high, 10))") is None


def _entry(formula: str, admitted: bool) -> dict:
    return {"factor_id": formula, "formula": formula, "admitted": admitted,
            "paper_ic": 0.05 if admitted else 0.01,
            "outcome": "admitted" if admitted else "low_ic"}


def test_insights_ignore_trait_shared_by_all_attempts() -> None:
    from factorminer.memory.formation import _derive_insights

    # Every candidate is non-linear, so admissions being non-linear says nothing.
    admitted = [_entry(f"IfElse($close, $open, {i})", True) for i in range(4)]
    rejected = [_entry(f"IfElse($high, $low, {i})", False) for i in range(6)]
    texts = [i.insight for i in _derive_insights(admitted, rejected, 1)]
    assert not any("non-linear" in t for t in texts)


def test_insights_report_rate_gap_with_both_sides_counted() -> None:
    from factorminer.memory.formation import _derive_insights

    admitted = [_entry(f"Skew($returns, {w})", True) for w in (5, 10, 20, 30)]
    rejected = [_entry(f"Skew($close, {w})", False) for w in (6, 12)]
    rejected += [_entry(f"Mean($close, {w})", False) for w in (3, 5, 8, 13, 21)]
    insights = _derive_insights(admitted, rejected, 2)
    nonlinear = [i for i in insights if "non-linear" in i.insight]
    assert len(nonlinear) == 1
    assert "4/6 (67%) with vs 0/5 (0%) without" in nonlinear[0].evidence


# ---------------------------------------------------------------------------
# Batch feedback -> retrieval -> prompt
# ---------------------------------------------------------------------------


def test_batch_feedback_separates_failure_kinds_and_names_collisions() -> None:
    feedback = summarize_trajectory(_trajectory(), iteration=4)
    assert feedback["counts"] == {"admitted": 1, "redundant": 1, "low_ic": 1, "parse_error": 1}
    assert feedback["near_misses"][0]["correlated_with"] == "#1 a"
    assert feedback["implementation_errors"][0]["reason"] == "Parse failure"

    text = render_batch_feedback(feedback)
    assert "FEEDBACK FROM LAST BATCH (iteration 4)" in text
    assert "rho 0.91 with #1 a" in text
    assert "Did not run" in text


def test_retrieval_prompt_includes_last_batch_feedback() -> None:
    memory = evolve_memory(ExperienceMemory(), form_memory(ExperienceMemory(), _trajectory(), 1))
    signal = retrieve_memory(memory, library_state={"library_size": 1})
    assert signal["batch_feedback"]["iteration"] == 1
    assert "FEEDBACK FROM LAST BATCH" in signal["prompt_text"]


def test_user_prompt_uses_library_state_keys_and_renders_memory_once() -> None:
    library = FactorLibrary(correlation_threshold=0.5, ic_threshold=0.04)
    library.admit_factor(Factor(
        id=0, name="f0", formula="Neg($close)", category="Other", ic_mean=0.05, icir=1.0,
        ic_win_rate=0.6, max_correlation=0.0, batch_number=1, ic_paper_mean=0.05,
    ))
    state = library.get_state_summary()
    state["target_size"] = 110
    memory = evolve_memory(ExperienceMemory(), form_memory(ExperienceMemory(), _trajectory(), 1))
    signal = retrieve_memory(memory, library_state=state)

    prompt = PromptBuilder().build_user_prompt(signal, state, batch_size=10)

    assert "Library size: 1 / 110 factors." in prompt
    assert "#1 Neg($close)" in prompt
    assert prompt.count("VWAP Deviation variants") == 1
    assert "{'name'" not in prompt


def test_structured_memory_fallback_formats_dict_entries() -> None:
    signal = {"recommended_directions": [{"name": "PV", "template": "Corr($close, $volume, 10)"}],
              "insights": [{"insight": "Use $amt", "evidence": "3/4 admitted"}]}
    prompt = PromptBuilder().build_user_prompt(signal, {"size": 3}, batch_size=5)
    assert "* PV -- template: Corr($close, $volume, 10)" in prompt
    assert "Note: Use $amt -- 3/4 admitted" in prompt
    assert "Library size: 3 / 110" in prompt


def test_parse_reflection_accepts_json_and_rejects_noise() -> None:
    raw = 'Sure:\n{"observations": ["x"], "hypotheses": "y", "avoid": []}'
    assert parse_reflection(raw) == {"observations": ["x"], "hypotheses": ["y"]}
    assert parse_reflection("1. alpha: Neg($close)") is None


def test_loop_passes_configured_reflection_budget(tmp_path, monkeypatch) -> None:
    received = []

    def reflect(provider, evidence, **kwargs):
        received.append(kwargs["max_tokens"])
        return {"observations": ["All candidates were weak."]}

    monkeypatch.setattr("factorminer.agent.reflection.reflect_on_batch", reflect)
    cfg = load_config(overrides={"memory": {"llm_reflection": True, "reflection_max_tokens": 8192}})
    cfg.output_dir = str(tmp_path)
    rng = np.random.default_rng(0)
    loop = RalphLoop(cfg, rng.normal(size=(5, 20, 8)), rng.normal(size=(5, 20)))
    loop.iteration = 1
    loop.memory.state.last_batch_feedback = {"iteration": 1, "outcomes": {"low_ic": 1}}
    loop._reflect_on_batch()
    assert received == [8192]
    assert loop.memory.state.last_batch_feedback["reflection"]["observations"]


# ---------------------------------------------------------------------------
# Config defaults that follow the paper protocol
# ---------------------------------------------------------------------------


def test_default_config_follows_paper_protocol() -> None:
    cfg = load_config()
    assert cfg.mining.icir_threshold == 0.0
    assert cfg.mining.train_split_only is True
    assert cfg.data.standardise_raw_features is False
    assert cfg.data.winsorise_raw_features is False
    assert cfg.memory.seed == "empty"
    assert cfg.memory.llm_reflection is False
    assert cfg.evaluation.redundancy_metric == "spearman"


def test_initial_memory_is_empty_unless_paper_defaults_requested(tmp_path) -> None:
    rng = np.random.default_rng(0)
    tensor, returns = rng.normal(size=(5, 20, 8)), rng.normal(size=(5, 20))
    cfg = load_config()
    cfg.output_dir = str(tmp_path / "a")
    assert RalphLoop(cfg, tensor, returns).memory.success_patterns == []

    seeded = load_config(overrides={"memory": {"seed": "paper_defaults"}})
    seeded.output_dir = str(tmp_path / "b")
    assert RalphLoop(seeded, tensor, returns).memory.success_patterns
