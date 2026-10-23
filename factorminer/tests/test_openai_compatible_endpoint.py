"""OpenAI-compatible endpoints: key source, reasoning output, request fields, tracing info."""

from __future__ import annotations

from types import SimpleNamespace

from factorminer.agent.llm_interface import (
    OpenAICompatibleProvider,
    create_provider,
    provider_call_info,
    split_thinking,
)
from factorminer.agent.output_parser import parse_llm_output
from factorminer.agent.provider_config import COMPAT_KEY_ENV, resolve_credential


class _FakeCompletions:
    def __init__(self, message, usage=None, finish_reason="stop"):
        self.calls = []
        self._response = SimpleNamespace(
            model="served-model",
            usage=usage,
            choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        )

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


def _provider_with(message, usage=None, **kwargs) -> tuple[OpenAICompatibleProvider, _FakeCompletions]:
    provider = OpenAICompatibleProvider(model="Qwen/Test", base_url="https://example.test/v1", **kwargs)
    completions = _FakeCompletions(message, usage)
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return provider, completions


def test_compat_key_comes_from_dedicated_env_var_only(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "hosted-key")
    monkeypatch.delenv(COMPAT_KEY_ENV, raising=False)
    assert resolve_credential("primary", "openai_compatible", {}).api_key == "local"
    monkeypatch.setenv(COMPAT_KEY_ENV, "compat-key")
    credential = resolve_credential("primary", "openai_compatible", {})
    assert credential.api_key == "compat-key"
    assert credential.source == f"env:{COMPAT_KEY_ENV}"
    # An explicit config key still wins; the draft role never reads it.
    assert resolve_credential("primary", "openai_compatible", {"api_key": "cfg"}).api_key == "cfg"
    assert resolve_credential("draft", "openai_compatible", {}).api_key == "local"


def test_split_thinking_variants() -> None:
    assert split_thinking("<think>plan</think>\n1. a: Neg($close)") == ("1. a: Neg($close)", "plan")
    assert split_thinking("plan only</think>answer") == ("answer", "plan only")
    assert split_thinking("<think>cut off at max_tokens") == ("", "cut off at max_tokens")
    assert split_thinking("no reasoning") == ("no reasoning", "")


def test_parser_ignores_formulas_drafted_inside_think() -> None:
    raw = "<think>1. draft: Neg($open)\n2. other: Neg($high)</think>\n1. final: Neg($close)"
    candidates, failed = parse_llm_output(raw)
    assert [c.name for c in candidates] == ["final"]
    assert failed == []


def test_generate_returns_answer_and_records_thinking_and_usage() -> None:
    message = SimpleNamespace(content="<think>compare windows</think>\n1. a: Neg($close)")
    usage = SimpleNamespace(prompt_tokens=900, completion_tokens=120, prompt_tokens_details=None)
    provider, _ = _provider_with(message, usage)
    assert provider.generate("sys", "user") == "1. a: Neg($close)"
    assert provider.last_call == {
        "model": "served-model",
        "usage": {"input": 900, "output": 120},
        "thinking": "compare windows",
    }


def test_reasoning_content_field_is_kept_as_thinking() -> None:
    message = SimpleNamespace(content="1. a: Neg($close)", reasoning_content="parser-separated")
    provider, _ = _provider_with(message)
    assert provider.generate("sys", "user") == "1. a: Neg($close)"
    assert provider.last_call["thinking"] == "parser-separated"


def test_extra_body_is_sent_with_each_request() -> None:
    message = SimpleNamespace(content="1. a: Neg($close)")
    extra = {"chat_template_kwargs": {"enable_thinking": False}}
    provider, completions = _provider_with(message, extra_body=extra)
    provider.generate("sys", "user", max_tokens=16384)
    assert completions.calls[0]["extra_body"] == extra
    assert completions.calls[0]["max_tokens"] == 16384


def test_guarded_provider_exposes_inner_call_info(monkeypatch) -> None:
    monkeypatch.setenv(COMPAT_KEY_ENV, "compat-key")
    guarded = create_provider({
        "provider": "openai_compatible",
        "model": "Qwen/Test",
        "base_url": "https://example.test/v1",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    })
    inner = guarded.inner
    assert inner.api_key == "compat-key"
    assert inner.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, prompt_tokens_details=None)
    inner._client = SimpleNamespace(chat=SimpleNamespace(
        completions=_FakeCompletions(SimpleNamespace(content="1. a: Neg($close)"), usage)
    ))
    guarded.generate(system_prompt="sys", user_prompt="user")
    assert provider_call_info(guarded)["usage"] == {"input": 10, "output": 5}


def test_neuroii_config_loads() -> None:
    from factorminer.cli.context import load_raw_config_data
    from factorminer.configs import DEFAULT_CONFIG_PATH
    from factorminer.utils.config import load_config

    path = DEFAULT_CONFIG_PATH.parent / "neuroii_qwen.yaml"
    cfg = load_config(path)
    raw = load_raw_config_data(str(path))
    assert cfg.llm.provider == "openai_compatible"
    assert cfg.llm.model == "Qwen/Qwen3.8-27B-FP8"
    assert raw["llm"]["base_url"] == "https://neuroii.com/v1"
    assert cfg.llm.max_tokens == 24576
    assert raw["llm"]["context_window"] == 32768


def test_max_tokens_shrinks_only_when_prompt_would_overflow_context() -> None:
    message = SimpleNamespace(content="1. a: Neg($close)")
    provider, completions = _provider_with(message, context_window=32768)
    provider.generate("s" * 8_500, "u" * 6_000, max_tokens=24576)  # typical prompt
    assert completions.calls[-1]["max_tokens"] == 24576
    provider.generate("s" * 8_500, "u" * 30_000, max_tokens=24576)  # oversized prompt
    fitted = completions.calls[-1]["max_tokens"]
    estimated_prompt = (8_500 + 30_000) * 2 // 5 + 64
    assert fitted < 24576
    assert fitted + estimated_prompt <= 32768


def test_no_context_window_keeps_requested_max_tokens() -> None:
    provider, completions = _provider_with(SimpleNamespace(content="x"))
    provider.generate("s" * 100_000, "u", max_tokens=24576)
    assert completions.calls[-1]["max_tokens"] == 24576
