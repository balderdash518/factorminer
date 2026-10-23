"""Managed templates must be the prompts actually sent and linked to generations."""

from __future__ import annotations

import json

import pytest

from factorminer.agent.factor_generator import FactorGenerator
from factorminer.agent.llm_interface import MockProvider
from factorminer.agent.prompt_builder import PromptBuilder
from factorminer.agent.prompt_management import (
    prompt_definitions,
    render_template,
    resolve_prompt,
)
from factorminer.agent.reflection import (
    REFLECTION_SYSTEM_PROMPT,
    build_reflection_prompt,
    reflect_on_batch,
)
from factorminer.tests.test_tracing import _FakeClient
from factorminer.utils.tracing import MiningTracer, activate
from scripts.sync_langfuse_prompts import same_template


class _Prompt:
    def __init__(self, definition):
        self.name = definition["name"]
        self.config = definition["config"]
        self.prompt = [dict(m) for m in definition["prompt"]]
        self.version = 7

    def compile(self, **variables):
        return [{**m, "content": render_template(m["content"], variables)} for m in self.prompt]


class _PromptClient(_FakeClient):
    def __init__(self, observation):
        super().__init__()
        self.prompt = _Prompt(prompt_definitions()[observation])

    def get_prompt(self, name, **kwargs):
        self.calls.append(("get_prompt", name, kwargs))
        return self.prompt


class _CapturingProvider(MockProvider):
    def generate(self, system_prompt, user_prompt, **kwargs):
        self.received = (system_prompt, user_prompt)
        return super().generate(system_prompt, user_prompt, **kwargs)


@pytest.mark.parametrize("observation,variables", [
    ("propose-factors", {"batch_context": "Generate exactly 1 candidate. {{literal}}"}),
    ("repair-formulas", {"broken_formulas": "  1. Negate($close)"}),
    ("reflect-on-batch", {"evidence": "low_ic: 2", "library_formulas": "  (empty)"}),
])
def test_managed_messages_match_local_templates(observation, variables, tmp_path, monkeypatch):
    definition = prompt_definitions()[observation]
    fake = _PromptClient(observation)
    monkeypatch.setattr(MiningTracer, "_connect", staticmethod(lambda *a, **k: fake))
    expected = fake.prompt.compile(**variables)
    with activate(MiningTracer(session_id="test", output_dir=tmp_path)):
        system, user, managed = resolve_prompt(
            observation, definition["prompt"][0]["content"], expected[1]["content"], variables,
        )
    assert (system, user) == (expected[0]["content"], expected[1]["content"])
    assert managed is fake.prompt
    assert fake.calls[0][2]["label"] == "production"


def test_generation_uses_remote_version_and_links_it(tmp_path, monkeypatch):
    fake = _PromptClient("propose-factors")
    fake.prompt.prompt[0]["content"] += "\nManaged instruction."
    monkeypatch.setattr(MiningTracer, "_connect", staticmethod(lambda *a, **k: fake))
    provider = _CapturingProvider()
    with activate(MiningTracer(session_id="test", output_dir=tmp_path)):
        FactorGenerator(provider).generate_batch(batch_size=1)
    assert provider.received[0].endswith("Managed instruction.")
    generation = next(c for c in fake.calls if c[:2] == ("start", "propose-factors"))
    assert generation[4]["prompt"] is fake.prompt
    assert generation[4]["input"][0]["content"] == provider.received[0]
    rows = [json.loads(s) for s in (tmp_path / "trace_log.jsonl").read_text().splitlines()]
    assert rows[0]["prompt"] == {"name": "factorminer/propose-factors", "version": 7}


def test_repair_and_reflection_link_their_own_templates(tmp_path, monkeypatch):
    fake = _PromptClient("repair-formulas")
    monkeypatch.setattr(MiningTracer, "_connect", staticmethod(lambda *a, **k: fake))
    provider = _CapturingProvider()
    with activate(MiningTracer(session_id="test", output_dir=tmp_path)):
        FactorGenerator(provider)._retry_failed_parses(["Negate($close)"], attempts=1)
        assert "  1. Negate($close)" in provider.received[1]
        repair = next(c for c in fake.calls if c[:2] == ("start", "repair-formulas"))
        assert repair[4]["prompt"].name == "factorminer/repair-formulas"
        fake.prompt = _Prompt(prompt_definitions()["reflect-on-batch"])
        reflect_on_batch(provider, "low_ic: 2", library_formulas=["Neg($close)"])
    assert provider.received == (
        REFLECTION_SYSTEM_PROMPT, build_reflection_prompt("low_ic: 2", ["Neg($close)"]),
    )
    reflection = next(c for c in fake.calls if c[:2] == ("start", "reflect-on-batch"))
    assert reflection[4]["prompt"].name == "factorminer/reflect-on-batch"


def test_outage_falls_back_without_repeated_fetches(tmp_path, monkeypatch):
    fake = _PromptClient("propose-factors")
    calls = []

    def unavailable(*args, **kwargs):
        calls.append(args)
        raise ConnectionError("offline")

    fake.get_prompt = unavailable
    monkeypatch.setattr(MiningTracer, "_connect", staticmethod(lambda *a, **k: fake))
    local = (PromptBuilder().system_prompt, "context", None)
    with activate(MiningTracer(session_id="test", output_dir=tmp_path)):
        for _ in range(2):
            assert resolve_prompt("propose-factors", *local[:2], {"batch_context": "context"}) == local
    assert len(calls) == 1


@pytest.mark.parametrize("bad", ["custom-system", "system-contract", "missing-variable", "extra-message"])
def test_incompatible_prompts_keep_local_context(bad, tmp_path, monkeypatch):
    fake = _PromptClient("propose-factors")
    system = PromptBuilder().system_prompt
    if bad == "custom-system":
        system = "Specialist/custom feature system."
    elif bad == "system-contract":
        fake.prompt.config = {"local_system_sha256": "old-registry"}
    elif bad == "missing-variable":
        fake.prompt.prompt[1]["content"] = "{{unknown}}"
    else:
        fake.prompt.prompt.append({"role": "assistant", "content": "example"})
    monkeypatch.setattr(MiningTracer, "_connect", staticmethod(lambda *a, **k: fake))
    with activate(MiningTracer(session_id="test", output_dir=tmp_path)):
        assert resolve_prompt("propose-factors", system, "local", {"batch_context": "local"}) == (
            system, "local", None,
        )


def test_management_can_be_disabled_without_fetching(tmp_path, monkeypatch):
    fake = _PromptClient("propose-factors")
    monkeypatch.setattr(MiningTracer, "_connect", staticmethod(lambda *a, **k: fake))
    with activate(MiningTracer(session_id="test", output_dir=tmp_path, prompt_management=False)):
        assert resolve_prompt("propose-factors", PromptBuilder().system_prompt, "local", {})[2] is None
    assert not any(c[0] == "get_prompt" for c in fake.calls)


def test_publishing_ignores_api_message_type_normalization():
    definition = prompt_definitions()["propose-factors"]
    server = {**definition, "prompt": [
        {k: v for k, v in m.items() if k != "type"} for m in definition["prompt"]
    ]}
    assert same_template(server, definition)
    server["prompt"][0]["content"] += "\nA real edit."
    assert not same_template(server, definition)
