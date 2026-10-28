"""Exercise the installed v4-compatible SDK without credentials or network export."""

from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest

from factorminer.utils.tracing import MiningTracer, activate
from scripts import sync_langfuse_prompts as prompts


def test_sdk_exports_root_io_hierarchy_and_session_on_every_generation(tmp_path):
    langfuse = pytest.importorskip("langfuse")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    client = langfuse.Langfuse(
        public_key=f"pk-lf-offline-{uuid4().hex}", secret_key="sk-lf-offline",
        base_url="https://example.invalid", tracer_provider=TracerProvider(),
        span_exporter=exporter, environment="client-default", release="migration-test",
    )
    tracer = MiningTracer(
        session_id="migration-session", output_dir=tmp_path, langfuse="off",
        environment="migration-check", tags=["migration-v4"],
        metadata={"campaign_id": "campaign-1", "escaped": '"\\\n' * 100},
    )
    tracer._client = client
    try:
        with activate(tracer), langfuse.propagate_attributes(environment="enclosing-scope"):
            with tracer.iteration(1, input={"batch_size": 2}) as root:
                with tracer.observe("generate-candidates", as_type="chain"):
                    with tracer.generation(
                        "propose-factors", model="mock", system_prompt="system",
                        user_prompt="user",
                    ) as generation:
                        generation.update(output="answer", usage_details={"input": 3, "output": 2})
                root.update(output={"admitted": ["factor"]})
            # A second iteration must be a new trace, with no context leakage.
            with tracer.iteration(2, input={"batch_size": 1}) as root:
                root.update(output={"admitted": []})
        spans = exporter.get_finished_spans()
        assert len(spans) == 4
        generation, chain, root, next_root = spans
        assert root.parent is None and next_root.parent is None
        assert generation.parent.span_id == chain.context.span_id
        assert chain.parent.span_id == root.context.span_id
        assert generation.context.trace_id == chain.context.trace_id == root.context.trace_id
        assert next_root.context.trace_id != root.context.trace_id
        assert json.loads(root.attributes["langfuse.observation.input"]) == {"batch_size": 2}
        assert json.loads(root.attributes["langfuse.observation.output"]) == {"admitted": ["factor"]}
        assert json.loads(generation.attributes["langfuse.observation.usage_details"]) == {
            "input": 3, "output": 2,
        }
        assert generation.attributes["langfuse.observation.output"] == "answer"
        for span in spans:
            attrs = span.attributes
            assert attrs["session.id"] == "migration-session"
            assert attrs["langfuse.trace.name"] == "mining-iteration"
            assert attrs["langfuse.trace.tags"] == ("migration-v4",)
            assert attrs["langfuse.environment"] == "migration-check"
            assert attrs["langfuse.release"] == "migration-test"
            assert json.loads(attrs["langfuse.trace.metadata.campaign_id"]) == "campaign-1"
            assert len(attrs["langfuse.trace.metadata.escaped"]) <= 200
            assert json.loads(attrs["langfuse.trace.metadata.escaped"]) == tracer.metadata["escaped"]
            assert "langfuse.trace.public" not in attrs
        assert json.loads(root.attributes["langfuse.trace.metadata.iteration"]) == "1"
        assert json.loads(next_root.attributes["langfuse.trace.metadata.iteration"]) == "2"
    finally:
        client.shutdown()


@pytest.mark.parametrize("body_fails", [False, True])
def test_exporter_exit_cannot_replace_mining_exception_or_break_local_mirror(tmp_path, body_fails):
    class BrokenClient:
        @contextmanager
        def start_as_current_observation(self, **fields):
            try:
                yield SimpleNamespace(update=lambda **fields: None)
            finally:
                raise RuntimeError("exporter teardown failed")

        def flush(self):
            raise RuntimeError("exporter flush failed")

    tracer = MiningTracer(session_id="local", output_dir=tmp_path, langfuse="off")
    tracer._client = BrokenClient()

    def run():
        with activate(tracer), tracer.observe("stage") as observation:
            observation.update(output="done")
            if body_fails:
                raise ValueError("mining failed")

    if body_fails:
        with pytest.raises(ValueError, match="mining failed"):
            run()
    else:
        run()
    record = json.loads((tmp_path / "trace_log.jsonl").read_text())
    assert record["output"] == "done"
    if body_fails:
        assert record["level"] == "ERROR"
        assert record["status_message"] == "ValueError: mining failed"


def test_conflicting_hosts_disable_remote_tracing_without_contacting_either_host(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-offline")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-offline")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("LANGFUSE_HOST", "https://other.invalid")
    assert MiningTracer._connect("on", "development", None, 1.0) is None


def test_remote_context_cannot_suppress_application_error(tmp_path):
    class SuppressingContext:
        def __enter__(self):
            return None

        def __exit__(self, *error):
            return True

    tracer = MiningTracer(session_id="local", output_dir=tmp_path, langfuse="off")
    tracer._client = SimpleNamespace(start_as_current_observation=lambda **fields: SuppressingContext())
    with pytest.raises(ValueError, match="application failed"):
        with tracer.observe("stage"):
            raise ValueError("application failed")


def test_observation_started_in_exception_handler_does_not_inherit_handled_error(tmp_path):
    exits = []

    class Context:
        def __enter__(self):
            return None

        def __exit__(self, *error):
            exits.append(error)

    tracer = MiningTracer(session_id="local", output_dir=tmp_path, langfuse="off")
    tracer._client = SimpleNamespace(start_as_current_observation=lambda **fields: Context())
    try:
        raise ValueError("already handled")
    except ValueError:
        with tracer.observe("fallback"):
            pass
    assert exits == [(None, None, None)]


def test_context_enter_failure_preserves_application_execution_and_local_output(tmp_path):
    class BrokenContext:
        def __enter__(self):
            raise RuntimeError("exporter enter failed")

        def __exit__(self, *error):
            pytest.fail("An unentered context must not be exited")

    tracer = MiningTracer(session_id="local", output_dir=tmp_path, langfuse="off")
    tracer._client = SimpleNamespace(start_as_current_observation=lambda **fields: BrokenContext())
    with tracer.observe("stage") as observation:
        observation.update(output="completed")
    assert json.loads((tmp_path / "trace_log.jsonl").read_text())["output"] == "completed"


@pytest.mark.parametrize("status,returncode", [(200, 0), (201, 0), (404, 5)])
def test_prompt_cli_uses_scoped_package_and_current_json_envelope(monkeypatch, status, returncode):
    calls = []
    monkeypatch.setattr(prompts.shutil, "which", lambda executable: "npx")

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(
            returncode=returncode, stdout=json.dumps({"status": status, "body": {"version": 1}}),
        )

    monkeypatch.setattr(prompts.subprocess, "run", run)
    assert prompts.request_cli(["get", "test-prompt"], {}) == (status, {"version": 1})
    assert calls == [["npx", "--yes", "@langfuse/cli@1.2.4", "api", "prompts",
                      "get", "test-prompt", "--json"]]


def test_prompt_cli_rejects_nonzero_exit_even_with_success_envelope(monkeypatch):
    monkeypatch.setattr(prompts.shutil, "which", lambda executable: "npx")
    monkeypatch.setattr(prompts.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=4, stdout='{"status":200,"body":{}}',
    ))
    with pytest.raises(RuntimeError, match="failed to complete"):
        prompts.request_cli(["get", "test-prompt"], {})
