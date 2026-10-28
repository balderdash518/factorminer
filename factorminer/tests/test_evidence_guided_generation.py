"""Offline contracts for the complete evidence-guided mining information flow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from factorminer.agent.factor_generator import FactorGenerator
from factorminer.agent.llm_interface import MockProvider
from factorminer.agent.output_parser import parse_llm_output
from factorminer.agent.prompt_builder import PromptBuilder
from factorminer.application.evidence_generation import (
    EvidenceGuidedGeneration,
    EvidenceGuidedSettings,
)
from factorminer.application.mining_budget import BudgetTracker, EvaluationResult
from factorminer.core.helix_loop import HelixLoop
from factorminer.core.ralph_loop import RalphLoop
from factorminer.memory.attempt_archive import AttemptArchive, formula_descriptor, library_identity
from factorminer.utils.config import load_config
from factorminer.utils.tracing import MiningTracer, activate


class ScriptedProvider(MockProvider):
    def __init__(self, batches=None, *, review_action="keep"):
        super().__init__()
        self.batches = list(batches or [])
        self.review_action = review_action
        self.prompts = []
        self.review_calls = 0

    def generate(self, system_prompt, user_prompt, **kwargs):
        self.prompts.append((system_prompt, user_prompt))
        self._call_count += 1
        self._last_call = {"model": "scripted-offline", "usage": {"input": 10, "output": 5}}
        if system_prompt.startswith("You review proposed alpha experiments"):
            self.review_calls += 1
            if self.review_action == "exception":
                raise ConnectionError("offline reviewer")
            if self.review_action == "bad":
                return "not a review"
            requests = json.loads(user_prompt)["candidates"]
            return json.dumps({"reviews": [{
                "attempt_id": r["attempt_id"],
                "action": "replace" if self.review_action == "unknown_evidence" else self.review_action,
                "reason": "Compare a different mechanism against the historical counterexample",
                "evidence_ids": ["invented"] if self.review_action == "unknown_evidence" else [r["history"][0]["attempt_id"]],
                "suggested_test": "Check a conditional mechanism, not an unplanned window sweep",
            } for r in requests]})
        return json.dumps(self.batches.pop(0) if self.batches else [
            {"name": "fallback", "formula": "Neg($close)"},
        ])


def _state(**updates):
    return {"members": [], "available_features": ["$close", "$open", "$volume", "$returns"], **updates}


def _guide(tmp_path, **settings):
    return EvidenceGuidedGeneration(
        EvidenceGuidedSettings(enabled=True, reserve_candidates=0, **settings), str(tmp_path), "test-scope",
    )


def _observe(archive, formula, *, outcome="low_ic", iteration=1, snapshot="", hypothesis=""):
    attempt_id = archive.propose("historical", formula, iteration, metadata={"hypothesis": hypothesis})
    archive.event(
        "evaluated", iteration, attempt_id=attempt_id, outcome=outcome,
        measured=outcome not in {"compute_error", "parse_error", "invalid_signal"},
        library_snapshot=snapshot, metrics={"ic_paper_mean": 0.01},
    )
    return attempt_id


def _filter(guide, pairs, provider=None, **kwargs):
    generator = FactorGenerator(provider or ScriptedProvider())
    budget = BudgetTracker()
    accepted, stats = guide.filter_batch(
        pairs, generator=generator, budget=budget, memory_signal={"evidence_guided": True},
        library_state=_state(), iteration=2, target_size=1, **kwargs,
    )
    return accepted, stats, budget


def test_exact_identity_preserves_executor_semantics_and_precision():
    def key(formula):
        return formula_descriptor(formula)["key"]
    assert key("Mean( $close, 5 )") == key("Mean($close,5)")
    assert key("Mean($close,5)") != key("Mean($close,10)")
    assert key("Log($returns)") != key("Log(Neg($returns))")
    assert key("Div($close,$volume)") != key("Mul($close,Inv($volume))")
    assert key("Add($close,1.0000001)") != key("Add($close,1.0000002)")
    assert key("Neg($close)") != key("$close")


def test_structured_proposals_preserve_declarations_and_legacy_format():
    text = json.dumps([{
        "formula": "Mean($close,5)", "name": "test", "hypothesis": "reversal",
        "reference_ids": ["a"], "declared_parent_id": "a",
        "change_description": "smooth", "expected_improvement": "less noise",
        "falsification_condition": "no change in stability",
    }])
    candidates, failed = parse_llm_output(text)
    assert not failed and candidates[0].is_valid
    assert candidates[0].evidence["declared_parent_id"] == "a"
    legacy, _ = parse_llm_output("1. old: Neg($close)")
    assert legacy[0].is_valid and legacy[0].evidence == {}


def test_archive_is_run_scoped_and_reloads_without_fabricating_measurements(tmp_path):
    archive = AttemptArchive(tmp_path, "run-a")
    a = archive.propose("not_tested", "Mean($close,5)", 1)
    archive.event("reviewed", 1, attempt_id=a, review={"action": "replace"})
    b = _observe(archive, "Std($returns,5)")
    loaded = AttemptArchive(tmp_path, "run-a")
    assert not loaded.attempts[a]["measured"]
    assert "metrics" not in loaded.attempts[a]
    assert loaded.attempts[b]["measured"]
    assert AttemptArchive(tmp_path, "run-b").attempts == {}


def test_retrieval_returns_explainable_parameter_edits_and_mixed_evidence(tmp_path):
    archive = AttemptArchive(tmp_path, "run")
    a = _observe(archive, "Mean($close,5)", outcome="admitted", hypothesis="price reversal")
    b = _observe(archive, "Mean($close,10)", hypothesis="price reversal")
    hits = archive.retrieve("price reversal", formula="Mean($close,20)", diverse=True)
    assert {h["attempt_id"] for h in hits} == {a, b}
    assert {h["role"] for h in hits} == {"success", "counterexample"}
    assert all(h["similarity"]["structural"] == 1 for h in hits)
    assert all(any("param:window" in d["path"] for d in h["differences"]) for h in hits)


def test_wrapper_edits_are_soft_neighbors_and_reviews_survive_retrieval(tmp_path):
    archive = AttemptArchive(tmp_path, "run")
    prior = _observe(archive, "Div($close,$volume)")
    archive.event("reviewed", 1, attempt_id=prior, review={
        "action": "revise", "reason": "Needs a purposeful comparison", "status": "advisory",
    })
    hits = archive.retrieve("price volume", formula="Neg(Div($close,$volume))")
    assert hits[0]["similarity"]["structural"] >= 0.65
    assert hits[0]["advisory_review"]["action"] == "revise"
    assert hits[0]["outcome"] == "low_ic"  # an opinion does not replace observation


def test_exact_repeats_are_replaced_before_evaluation(tmp_path):
    guide = _guide(tmp_path)
    prior_id = _observe(guide.archive, "Neg($close)")
    provider = ScriptedProvider([[{"name": "new", "formula": "Std($returns,5)"}]])
    accepted, stats, budget = _filter(guide, [("old", "Neg($close)")], provider)
    assert accepted == [("new", "Std($returns, 5)")]
    assert stats["generation_duplicates"] == 1
    assert stats["generation_replacement_rounds"] == 1
    assert budget.llm_calls == 1
    assert prior_id in provider.prompts[0][1]
    assert '"ic_paper_mean": 0.01' in provider.prompts[0][1]
    rejected = [r for r in guide.archive.attempts.values() if r.get("status") == "already_tested"]
    assert rejected and all(not r["measured"] for r in rejected)


def test_replacements_are_bounded_and_partial_batches_are_allowed(tmp_path):
    guide = _guide(tmp_path, max_replacement_rounds=2)
    _observe(guide.archive, "Neg($close)")
    accepted, stats, budget = _filter(guide, [("old", "Neg($close)")])
    assert accepted == []
    assert stats["generation_replacement_rounds"] == 2
    assert stats["generation_shortfall"] == 1
    assert budget.llm_calls == 2


@pytest.mark.parametrize("action", ["keep", "bad", "exception", "unknown_evidence"])
def test_similarity_or_failed_reviewer_never_becomes_a_hard_veto(tmp_path, action):
    guide = _guide(tmp_path, max_replacement_rounds=0)
    _observe(guide.archive, "Mean($close,5)")
    provider = ScriptedProvider(review_action=action)
    accepted, stats, budget = _filter(guide, [("candidate", "Mean($close,10)")], provider)
    assert accepted == [("candidate", "Mean($close,10)")]
    assert provider.review_calls == 1
    assert stats["generation_similar"] == 1
    assert budget.llm_calls == 1
    record = guide.archive.attempts[next(iter(guide.pending.values()))]
    assert record["review"]["status"] == ("advisory" if action == "keep" else "abstained")


def test_reviewer_revision_returns_to_generator_and_is_checked_again(tmp_path):
    guide = _guide(tmp_path)
    _observe(guide.archive, "Mean($close,5)")
    provider = ScriptedProvider([
        [{"name": "different", "formula": "Std($returns,5)", "hypothesis": "volatility"}],
    ], review_action="revise")
    accepted, stats, budget = _filter(guide, [("near", "Mean($close,10)")], provider)
    assert accepted == [("different", "Std($returns, 5)")]
    assert stats["generation_revision_requests"] == 1
    assert budget.llm_calls == 2
    assert "suggested_test" in provider.prompts[-1][1]


def test_generator_can_retain_a_purposeful_control_but_not_override_exact_check(tmp_path):
    guide = _guide(tmp_path)
    _observe(guide.archive, "Mean($close,5)")
    provider = ScriptedProvider([[{
        "name": "control", "formula": "Mean($close,10)",
        "retention_reason": "Predefined window stability control",
        "falsification_condition": "Stability does not improve",
    }]], review_action="revise")
    accepted, _, _ = _filter(guide, [("near", "Mean($close,10)")], provider)
    assert accepted == [("control", "Mean($close, 10)")]
    assert provider.review_calls == 1
    _observe(guide.archive, "Mean($close,10)")
    accepted, _, _ = _filter(guide, [("near", "Mean($close,10)")], provider, allow_replacement=False)
    assert accepted == []


def test_execution_failure_can_retry_but_library_dependent_failure_requires_same_snapshot(tmp_path):
    guide = _guide(tmp_path, review_enabled=False, max_replacement_rounds=0)
    _observe(guide.archive, "Mean($close,5)", outcome="compute_error")
    accepted, _, _ = _filter(guide, [("retry", "Mean($close,5)")])
    assert accepted
    current = library_identity(_state())
    prior_id = _observe(guide.archive, "Std($returns,5)", outcome="redundant", snapshot="old-library")
    key = formula_descriptor("Std($returns,5)")["key"]
    assert guide.archive.exact_observation(key, current) is None
    guide.archive.event("evaluated", 1, attempt_id=prior_id, library_snapshot=current)
    assert guide.archive.exact_observation(key, current) is not None


def test_context_budget_and_trace_show_only_selected_evidence(tmp_path):
    guide = _guide(tmp_path, max_context_chars=2500)
    for window in range(3, 18):
        _observe(guide.archive, f"Mean($close,{window})", hypothesis="price reversal")
    with activate(MiningTracer(session_id="offline", output_dir=tmp_path, langfuse="off")):
        context = guide.prepare(
            {"recommended_directions": ["price reversal"]}, _state(), iteration=2,
            last_feedback={}, family_ledger={}, research_action={},
        )
    assert len(context["attempt_history_text"]) <= 2500
    shown = context["attempt_history"]
    assert shown and len(shown) < 6
    prompt = PromptBuilder().build_user_prompt(context, _state(), 2)
    assert all(h["attempt_id"] in prompt for h in shown)
    trace = [json.loads(s) for s in (tmp_path / "trace_log.jsonl").read_text(encoding="utf-8").splitlines()]
    retrieval = next(r for r in trace if r["name"] == "retrieve-attempt-history")
    assert retrieval["output"]["selected"] == shown


def test_context_budget_holds_with_large_existing_feedback(tmp_path):
    guide = _guide(tmp_path, max_context_chars=2000)
    context = guide.prepare(
        {"recommended_directions": ["long " * 3000]}, _state(), iteration=2,
        last_feedback={"description": "feedback " * 3000},
        family_ledger={"family": "examples " * 3000}, research_action={},
    )
    assert len(context["attempt_history_text"]) <= 2000


def test_unmeasured_execution_failures_do_not_get_zero_ic_evidence(tmp_path):
    guide = _guide(tmp_path, max_replacement_rounds=0, review_enabled=False)
    _filter(guide, [("failed", "Neg($close)")])
    result = EvaluationResult(
        factor_name="failed", formula="Neg($close)", parse_ok=True,
        outcome="compute_error", rejection_reason="computation unavailable",
    )
    guide.record_results([result], iteration=2)
    record = guide.archive.attempts[result.attempt_id]
    assert not record["measured"]
    assert record["metrics"]["ic_paper_mean"] is None
    assert guide.archive.summary()["measured"] == 0


def test_explicit_parent_and_inferred_neighbors_are_separate(tmp_path):
    guide = _guide(tmp_path, max_replacement_rounds=0, review_enabled=False)
    parent = _observe(guide.archive, "Mean($close,5)")
    provider = ScriptedProvider([[{
        "name": "child", "formula": "Mean($close,10)", "declared_parent_id": parent,
        "hypothesis": "Controlled smoothing comparison", "reference_ids": [parent],
    }]])
    generator = FactorGenerator(provider)
    candidates = generator.generate_batch(memory_signal={"evidence_guided": True}, batch_size=1)
    guide.filter_batch(
        [(c.name, c.formula) for c in candidates], generator=generator, budget=BudgetTracker(),
        memory_signal={"evidence_guided": True}, library_state=_state(), iteration=2, target_size=1,
    )
    result = EvaluationResult(factor_name="child", formula=candidates[0].formula, parse_ok=True, outcome="low_ic")
    guide.record_results([result], iteration=2)
    assert result.declared_parent_id == parent and result.nearest_prior_ids == [parent]
    assert result.parent_formula == ""  # retrieval never invents an edit edge


def test_repeated_batch_and_missing_features_do_not_reach_evaluation(tmp_path):
    guide = _guide(tmp_path, max_replacement_rounds=0, review_enabled=False)
    pairs = [("first", "Mean($close,5)"), ("same", "Mean( $close, 5 )"), ("missing", "Mean($vwap,5)")]
    accepted, stats, _ = _filter(guide, pairs)
    assert accepted == pairs[:1]
    assert stats["generation_duplicates"] == stats["generation_invalid"] == 1


@pytest.mark.parametrize("loop_class", [RalphLoop, HelixLoop])
def test_full_loop_archives_final_results_and_feeds_next_generation(tmp_path, loop_class):
    cfg = load_config(overrides={
        "mining": {"batch_size": 1, "target_library_size": 20},
        "evaluation": {"num_workers": 1},
        "memory": {"evidence_guided": {"enabled": True, "reserve_candidates": 0, "review_enabled": False}},
    })
    cfg.output_dir = str(tmp_path)
    rng = np.random.default_rng(8)
    provider = ScriptedProvider([
        [{"name": "first", "formula": "Neg($close)", "hypothesis": "reversal"}],
        [{"name": "repeat", "formula": "Neg($close)"}],
        [{"name": "next", "formula": "Std($returns,5)", "hypothesis": "volatility"}],
    ])
    loop = loop_class(cfg, rng.normal(size=(10, 45, 8)), rng.normal(size=(10, 45)), llm_provider=provider)
    loop.iteration = 1
    loop._run_iteration(1)
    prior = next(r for r in loop.evidence_guidance.archive.attempts.values() if r.get("status") == "evaluated")
    loop.iteration = 2
    stats = loop._run_iteration(1)
    assert stats["generation_duplicates"] == 1
    assert prior["attempt_id"] in provider.prompts[1][1]
    observed = [r for r in loop.evidence_guidance.archive.attempts.values() if r.get("status") == "evaluated"]
    assert len(observed) == 2
    assert loop.budget.llm_calls == len(provider.prompts)
    assert all(r.get("library_snapshot") for r in observed)


def test_enhanced_loop_does_not_infer_parent_edges_from_library_similarity(tmp_path):
    cfg = load_config(overrides={"memory": {"evidence_guided": {"enabled": True}}})
    cfg.output_dir = str(tmp_path)
    rng = np.random.default_rng(2)
    loop = RalphLoop(cfg, rng.normal(size=(5, 20, 8)), rng.normal(size=(5, 20)), llm_provider=ScriptedProvider())
    _filter(loop.evidence_guidance, [("near", "Mean($close,10)")])
    result = EvaluationResult(factor_name="near", formula="Mean($close,10)")
    loop._annotate_result_lineage([result], {"members": [{"formula": "Mean($close,5)", "ic_paper_mean": 0.1}]})
    assert result.parent_formula == "" and result.declared_parent_id == ""


def test_checkpoint_restores_run_history_and_does_not_overwrite_new_output(tmp_path):
    guide = _guide(tmp_path / "old")
    prior = _observe(guide.archive, "Neg($close)")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    state = guide.save_checkpoint(checkpoint)
    restored = _guide(tmp_path / "new")
    foreign = restored.archive.propose("foreign", "Std($returns,5)", 1)
    restored.restore_checkpoint(checkpoint, state)
    assert prior in restored.archive.attempts and foreign not in restored.archive.attempts
    rows_before = restored.archive.path.read_text(encoding="utf-8").splitlines()
    restored.restore_checkpoint(checkpoint, state)
    assert restored.archive.path.read_text(encoding="utf-8").splitlines() == rows_before
    assert any(foreign in row for row in rows_before)


def test_old_checkpoint_cannot_read_later_results_from_same_output_directory(tmp_path):
    guide = _guide(tmp_path / "run")
    prior = _observe(guide.archive, "Neg($close)")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    state = guide.save_checkpoint(checkpoint)
    future = _observe(guide.archive, "Std($returns,5)", iteration=10)
    restored = _guide(tmp_path / "run")
    restored.restore_checkpoint(checkpoint, state)
    assert prior in restored.archive.attempts and future not in restored.archive.attempts
    next_checkpoint = tmp_path / "checkpoint2"
    next_checkpoint.mkdir()
    restored.save_checkpoint(next_checkpoint)
    assert future not in (next_checkpoint / "attempt_archive.jsonl").read_text(encoding="utf-8")


def test_full_run_traces_review_under_its_agent_and_counts_each_call(tmp_path, monkeypatch):
    from factorminer.tests.test_tracing import _FakeClient

    fake = _FakeClient()
    monkeypatch.setattr(MiningTracer, "_connect", staticmethod(lambda *a, **k: fake))
    cfg = load_config(overrides={
        "mining": {"batch_size": 1, "target_library_size": 20},
        "evaluation": {"num_workers": 1},
        "memory": {"evidence_guided": {"enabled": True, "reserve_candidates": 0}},
    })
    cfg.output_dir = str(tmp_path)
    rng = np.random.default_rng(4)
    provider = ScriptedProvider([[{"name": "near", "formula": "Mean($close,10)"}]], review_action="keep")
    loop = RalphLoop(cfg, rng.normal(size=(10, 45, 8)), rng.normal(size=(10, 45)), llm_provider=provider)
    _observe(loop.evidence_guidance.archive, "Mean($close,5)")
    loop.run(max_iterations=1, resume=False)
    starts = {c[1]: c for c in fake.calls if c[0] == "start"}
    assert starts["retrieve-attempt-history"][2:4] == ("retriever", "mining-iteration")
    assert starts["review-candidate-experiments"][2:4] == ("agent", "check-candidate-history")
    assert starts["review-similar-attempts"][2:4] == ("generation", "review-candidate-experiments")
    assert loop.budget.llm_calls == 2
    assert loop.budget.total_tokens == 30
    trace = [json.loads(s) for s in (tmp_path / "trace_log.jsonl").read_text(encoding="utf-8").splitlines()]
    review = next(r for r in trace if r["name"] == "review-similar-attempts")
    assert json.loads(review["input"][1]["content"])["candidates"][0]["history"]
    assert review["usage_details"] == {"input": 10, "output": 5}
    assert "attempt_archive" in loop._run_manifest["artifact_paths"]


def test_all_repeats_return_a_logged_partial_batch_without_evaluating(tmp_path, monkeypatch):
    cfg = load_config(overrides={
        "memory": {"evidence_guided": {"enabled": True, "reserve_candidates": 0}},
    })
    cfg.output_dir = str(tmp_path)
    rng = np.random.default_rng(6)
    provider = ScriptedProvider()
    loop = RalphLoop(cfg, rng.normal(size=(5, 20, 8)), rng.normal(size=(5, 20)), llm_provider=provider)
    _observe(loop.evidence_guidance.archive, "Neg($close)")

    def unexpected_evaluation(*args):
        raise AssertionError("No approved candidates should contact the evaluator")

    monkeypatch.setattr(loop.pipeline, "evaluate_batch", unexpected_evaluation)
    loop.iteration = 1
    stats = loop._run_iteration(1)
    assert stats["generation_shortfall"] == 1 and stats["generation_duplicates"] == 3
    assert loop.budget.llm_calls == 3  # first generation plus two replacements
    assert (tmp_path / "mining_batches.jsonl").exists()


def test_loop_checkpoint_restores_history_in_a_new_directory(tmp_path):
    cfg = load_config(overrides={"memory": {"evidence_guided": {"enabled": True}}})
    cfg.output_dir = str(tmp_path / "old")
    rng = np.random.default_rng(3)
    data, returns = rng.normal(size=(5, 20, 8)), rng.normal(size=(5, 20))
    loop = RalphLoop(cfg, data, returns, llm_provider=ScriptedProvider())
    prior = _observe(loop.evidence_guidance.archive, "Neg($close)")
    saved = loop.save_session()
    cfg.output_dir = str(tmp_path / "new")
    resumed = RalphLoop(cfg, data, returns, llm_provider=ScriptedProvider())
    resumed.load_session(saved)
    assert prior in resumed.evidence_guidance.archive.attempts


def test_new_run_does_not_implicitly_inherit_previous_run(tmp_path):
    first = _guide(tmp_path)
    _observe(first.archive, "Neg($close)")
    second = _guide(tmp_path)
    assert not second.archive.attempts


@pytest.mark.parametrize("settings", [
    {"max_replacement_rounds": -1}, {"similarity_threshold": 2},
    {"enabled": "yes"}, {"history_limit": 0}, {"max_context_chars": 100},
    {"unknown_setting": True},
])
def test_config_validation_rejects_invalid_settings(settings):
    with pytest.raises(ValueError):
        load_config(overrides={"memory": {"evidence_guided": settings}})


def test_feature_is_off_for_paper_baseline_and_has_runnable_sample():
    assert not load_config().memory.evidence_guided["enabled"]
    root = Path(__file__).resolve().parents[2]
    sample = load_config(root / "factorminer/configs/evidence_guided.yaml")
    assert sample.memory.evidence_guided["enabled"] and sample.llm.provider == "mock"
