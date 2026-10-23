"""Run tracing: no-op default, local mirror, Langfuse span structure, token accounting."""

from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from factorminer.agent.llm_interface import (
    MockProvider,
    _anthropic_call_info,
    _openai_call_info,
)
from factorminer.core.ralph_loop import RalphLoop
from factorminer.utils.config import load_config
from factorminer.utils.tracing import MiningTracer, activate, get_tracer

STAGES = [
    "retrieve-memory",
    "propose-factors",
    "generate-candidates",
    "evaluate-candidates",
    "update-library",
    "distill-memory",
    "mining-iteration",
]


def _loop(tmp_path, provider=None, **mining) -> RalphLoop:
    rng = np.random.default_rng(0)
    tensor = rng.normal(size=(12, 40, 8))
    returns = rng.normal(0, 0.02, (12, 40))
    cfg = load_config(overrides={"mining": {"batch_size": 6, "max_iterations": 1, **mining}})
    cfg.output_dir = str(tmp_path)
    return RalphLoop(cfg, tensor, returns, llm_provider=provider or MockProvider())


@pytest.fixture(autouse=True)
def _no_langfuse_credentials(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)


def test_default_tracer_is_noop_outside_a_run() -> None:
    tracer = get_tracer()
    assert tracer.langfuse_enabled is False
    with tracer.observe("anything") as obs:
        obs.update(output={"x": 1})


def test_langfuse_stays_off_without_credentials(tmp_path) -> None:
    assert not MiningTracer(session_id="s", output_dir=tmp_path).langfuse_enabled
    assert not MiningTracer(session_id="s", output_dir=tmp_path, langfuse="on").langfuse_enabled


def test_mining_run_writes_local_trace_mirror(tmp_path) -> None:
    loop = _loop(tmp_path)
    loop.run(max_iterations=1, resume=False)

    rows = [
        json.loads(line)
        for line in (tmp_path / "trace_log.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [r["name"] for r in rows] == STAGES
    by_name = {r["name"]: r for r in rows}
    generation = by_name["propose-factors"]
    assert generation["type"] == "generation"
    assert generation["input"][1]["role"] == "user"
    assert "Generate exactly 6" in generation["input"][1]["content"]
    ref = generation["input"][0]["content_ref"]
    assert (tmp_path / ref).read_text(encoding="utf-8").startswith("You are a quantitative")
    evaluation = by_name["evaluate-candidates"]["output"]
    assert sum(evaluation["outcomes"].values()) == len(evaluation["candidates"])
    assert {"outcome", "paper_ic", "max_correlation", "reason"} <= set(evaluation["candidates"][0])
    assert "batch_feedback" in by_name["distill-memory"]["output"]
    assert all(r["session_id"].startswith("factorminer-") for r in rows)


def test_local_mirror_can_be_disabled(tmp_path) -> None:
    loop = _loop(tmp_path)
    loop.config.observability.local_log = False
    loop.run(max_iterations=1, resume=False)
    assert not (tmp_path / "trace_log.jsonl").exists()


class _FakeObservation:
    def __init__(self, calls, name):
        self.calls, self.name = calls, name

    def update(self, **fields):
        self.calls.append(("update", self.name, fields))


class _FakeClient:
    """Records what MiningTracer sends to the Langfuse client."""

    def __init__(self):
        self.calls: list = []
        self.stack: list[str] = []

    @contextmanager
    def start_as_current_observation(self, *, name, as_type, **fields):
        self.calls.append(("start", name, as_type, self.stack[-1] if self.stack else None, fields))
        self.stack.append(name)
        try:
            yield _FakeObservation(self.calls, name)
        finally:
            self.stack.pop()

    def get_current_trace_id(self):
        return "trace-1"

    def create_score(self, **fields):
        self.calls.append(("score", fields["name"], fields))

    def flush(self):
        self.calls.append(("flush",))


def test_langfuse_calls_nest_stages_and_score_the_iteration(tmp_path, monkeypatch) -> None:
    fake = _FakeClient()
    monkeypatch.setattr(MiningTracer, "_connect", staticmethod(lambda *a, **k: fake))
    _loop(tmp_path).run(max_iterations=1, resume=False)

    starts = {c[1]: c for c in fake.calls if c[0] == "start"}
    assert starts["mining-iteration"][2:4] == ("agent", None)
    assert starts["retrieve-memory"][2:4] == ("retriever", "mining-iteration")
    assert starts["generate-candidates"][2:4] == ("chain", "mining-iteration")
    assert starts["propose-factors"][2:4] == ("generation", "generate-candidates")
    assert starts["evaluate-candidates"][2:4] == ("evaluator", "mining-iteration")
    assert starts["propose-factors"][4]["input"][0]["role"] == "system"

    scores = {c[1]: c[2] for c in fake.calls if c[0] == "score"}
    assert {"yield_rate", "redundancy_rate", "weak_signal_rate",
            "implementation_failure_rate", "best_paper_ic", "library_size"} <= set(scores)
    assert scores["yield_rate"]["trace_id"] == "trace-1"
    assert fake.calls[-1] == ("flush",)


class _UsageProvider(MockProvider):
    def generate(self, *args, **kwargs):
        text = super().generate(*args, **kwargs)
        self._last_call = {"model": "gpt-test", "usage": {"input": 120, "output": 30}}
        return text


def test_generator_tokens_reach_budget_and_generation(tmp_path, monkeypatch) -> None:
    fake = _FakeClient()
    monkeypatch.setattr(MiningTracer, "_connect", staticmethod(lambda *a, **k: fake))
    loop = _loop(tmp_path, provider=_UsageProvider())
    loop.run(max_iterations=1, resume=False)

    assert loop.budget.llm_prompt_tokens == 120
    assert loop.budget.llm_completion_tokens == 30
    gen_updates = [c[2] for c in fake.calls if c[0] == "update" and c[1] == "propose-factors"]
    assert gen_updates[-1]["usage_details"] == {"input": 120, "output": 30}
    assert gen_updates[-1]["model"] == "gpt-test"


def test_openai_usage_splits_cached_tokens() -> None:
    response = SimpleNamespace(
        model="gpt-x",
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            prompt_tokens_details=SimpleNamespace(cached_tokens=30),
        ),
    )
    assert _openai_call_info("gpt-x", response) == {
        "model": "gpt-x",
        "usage": {"input": 70, "output": 20, "input_cached_tokens": 30},
    }


def test_anthropic_call_info_keeps_thinking_and_cache_reads() -> None:
    response = SimpleNamespace(
        model="claude-test",
        usage=SimpleNamespace(input_tokens=50, output_tokens=10, cache_read_input_tokens=400),
        content=[
            SimpleNamespace(type="thinking", thinking="compare windows first"),
            SimpleNamespace(type="text", text="1. a: Neg($close)"),
        ],
    )
    info = _anthropic_call_info("claude-test", response)
    assert info["usage"] == {"input": 50, "output": 10, "cache_read_input_tokens": 400}
    assert info["thinking"] == "compare windows first"


def test_exporter_failures_never_break_mining(tmp_path, monkeypatch) -> None:
    class _Broken(_FakeClient):
        def start_as_current_observation(self, **_):
            raise RuntimeError("exporter down")

        def create_score(self, **_):
            raise RuntimeError("exporter down")

    monkeypatch.setattr(MiningTracer, "_connect", staticmethod(lambda *a, **k: _Broken()))
    loop = _loop(tmp_path)
    loop.run(max_iterations=1, resume=False)
    assert loop.iteration == 1


def test_activate_restores_previous_tracer(tmp_path) -> None:
    tracer = MiningTracer(session_id="s", output_dir=tmp_path, langfuse="off")
    with activate(tracer):
        assert get_tracer() is tracer
    assert get_tracer() is not tracer
