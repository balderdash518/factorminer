"""Run tracing for the mining loop: Langfuse when configured, local JSONL always.

Each mining iteration is one Langfuse trace (an ``agent`` observation named
``mining-iteration``); all iterations of a run share one ``session_id`` so the
Sessions view shows the whole campaign in order. Inside an iteration:

* ``retrieve-memory`` (retriever) -- the memory text the generator will see;
* ``generate-candidates`` (chain) with one ``generation`` per LLM call
  (``propose-factors``, ``repair-formulas``) carrying model, prompts, raw
  output, token usage and, when the provider returns it, thinking text;
* ``evaluate-candidates`` (evaluator) -- every candidate's outcome and stats;
* ``update-library`` (span) and ``distill-memory`` (span, plus an optional
  ``reflect-on-batch`` generation).

Iteration-level numbers (yield, redundancy rate, parse-failure rate, ...) are
attached as trace scores so dashboards can show where the framework loses
candidates over a run.

Langfuse is optional: without the ``langfuse`` package or credentials every
call is a no-op. The local mirror (``trace_log.jsonl`` in the run directory)
records the same observations so a run can be reviewed offline; system prompts
are stored once per distinct text in ``system_prompts/``.

Tracing must never change or break mining: every exporter error is logged and
swallowed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MAX_METADATA_VALUE = 200  # SDK v5 limits the JSON-encoded propagated value


@contextmanager
def _remote_context(context: Any) -> Iterator[Any]:
    """Keep exporter context failures from replacing or suppressing mining errors."""
    try:
        remote = context.__enter__()
    except Exception as exc:
        logger.debug("Langfuse context enter failed: %s", exc)
        yield None
        return
    error: tuple[Any, Any, Any] = (None, None, None)
    try:
        yield remote
    except BaseException:
        error = sys.exc_info()
        raise
    finally:
        try:
            context.__exit__(*error)
        except Exception as exc:
            logger.debug("Langfuse context exit failed: %s", exc)


class _NullObservation:
    """Stand-in returned when Langfuse is off; keeps call sites branch-free."""

    def update(self, **_: Any) -> None:
        return None


class _Observation:
    """One open observation: forwards updates to Langfuse and the local mirror."""

    def __init__(self, tracer: MiningTracer, record: dict[str, Any], remote: Any) -> None:
        self._tracer = tracer
        self._record = record
        self._remote = remote

    def update(self, **fields: Any) -> None:
        for key, value in fields.items():
            if value is not None:
                self._record[key] = value
        if self._remote is not None:
            try:
                self._remote.update(**fields)
            except Exception as exc:  # pragma: no cover - exporter failures never break mining
                logger.debug("Langfuse update failed: %s", exc)


class MiningTracer:
    """Traces one mining run. Use :func:`get_tracer` from inside the loop."""

    def __init__(
        self,
        *,
        session_id: str,
        output_dir: str | Path | None,
        langfuse: str = "auto",
        local_log: bool = True,
        environment: str = "development",
        release: str | None = None,
        tags: list[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        sample_rate: float = 1.0,
        prompt_management: bool = True,
    ) -> None:
        self.session_id = _ascii(session_id)[:200]
        self.tags = sorted({_ascii(tag)[:200] for tag in (tags or []) if tag})
        self.metadata = {
            _ascii(str(key)): _short(value) for key, value in dict(metadata or {}).items()
        }
        self.environment = environment
        self._prompt_management = prompt_management
        self._prompt_failures: set[str] = set()
        self._iteration: int | None = None
        self._trace_id: str | None = None
        self._local_path: Path | None = None
        self._prompt_dir: Path | None = None
        if local_log and output_dir is not None:
            self._local_path = Path(output_dir) / "trace_log.jsonl"
            self._prompt_dir = Path(output_dir) / "system_prompts"
            self._local_path.parent.mkdir(parents=True, exist_ok=True)
        self._client = self._connect(langfuse, environment, release, sample_rate)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @staticmethod
    def _connect(mode: str, environment: str, release: str | None, sample_rate: float) -> Any:
        mode = str(mode or "auto").lower()
        if mode == "off":
            return None
        has_keys = bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))
        if not has_keys:
            if mode == "on":
                logger.warning(
                    "observability.langfuse=on but LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY "
                    "are not set; Langfuse tracing disabled"
                )
            return None
        try:
            from langfuse import Langfuse
        except ImportError:
            logger.warning(
                "Langfuse credentials found but the langfuse package is not installed "
                "(pip install 'factorminer[observability]'); Langfuse tracing disabled"
            )
            return None
        try:
            base_url = os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST")
            host = os.environ.get("LANGFUSE_HOST")
            if base_url and host and base_url.rstrip("/") != host.rstrip("/"):
                raise ValueError("LANGFUSE_BASE_URL and LANGFUSE_HOST must identify the same host")
            return Langfuse(
                base_url=base_url,
                environment=environment,
                release=release,
                sample_rate=sample_rate,
            )
        except Exception as exc:
            logger.warning("Langfuse client initialisation failed; tracing disabled: %s", exc)
            return None

    @property
    def langfuse_enabled(self) -> bool:
        return self._client is not None

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    @contextmanager
    def iteration(self, iteration: int, *, input: Any = None) -> Iterator[Any]:
        """Root observation for one mining iteration (one trace)."""
        self._iteration = int(iteration)
        with ExitStack() as stack:
            if self._client is not None:
                try:
                    from langfuse import propagate_attributes

                    stack.enter_context(
                        _remote_context(
                            propagate_attributes(
                                session_id=self.session_id,
                                trace_name="mining-iteration",
                                tags=self.tags,
                                metadata={**self.metadata, "iteration": str(iteration)},
                                environment=self.environment,
                            )
                        )
                    )
                except Exception as exc:  # pragma: no cover
                    logger.debug("Langfuse propagate_attributes failed: %s", exc)
            with self.observe("mining-iteration", as_type="agent", input=input) as root:
                if self._client is not None:
                    try:
                        self._trace_id = self._client.get_current_trace_id()
                    except Exception:  # pragma: no cover
                        self._trace_id = None
                try:
                    yield root
                finally:
                    self._trace_id = None

    @contextmanager
    def observe(
        self,
        name: str,
        *,
        as_type: str = "span",
        input: Any = None,
        metadata: Mapping[str, Any] | None = None,
        **generation_fields: Any,
    ) -> Iterator[Any]:
        """Open a child observation under whatever observation is current."""
        record: dict[str, Any] = {
            "session_id": self.session_id,
            "iteration": self._iteration,
            "name": name,
            "type": as_type,
            "start": _now(),
        }
        if input is not None:
            record["input"] = input
        if metadata:
            record["metadata"] = dict(metadata)
        for key in ("model", "model_parameters"):
            if generation_fields.get(key) is not None:
                record[key] = generation_fields[key]
        prompt = generation_fields.get("prompt")
        if prompt is not None:
            record["prompt"] = {"name": prompt.name, "version": prompt.version}

        remote_cm = None
        if self._client is not None:
            try:
                remote_cm = self._client.start_as_current_observation(
                    name=name,
                    as_type=as_type,
                    input=input,
                    metadata=dict(metadata) if metadata else None,
                    **{k: v for k, v in generation_fields.items() if v is not None},
                )
            except Exception as exc:  # pragma: no cover
                logger.debug("Langfuse start_as_current_observation failed: %s", exc)
                remote_cm = None

        started = time.perf_counter()
        with ExitStack() as stack:
            remote = None
            if remote_cm is not None:
                try:
                    remote = stack.enter_context(_remote_context(remote_cm))
                except Exception as exc:  # pragma: no cover
                    logger.debug("Langfuse observation enter failed: %s", exc)
            observation = _Observation(self, record, remote)
            try:
                yield observation
            except BaseException as exc:
                observation.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                record["duration_s"] = round(time.perf_counter() - started, 4)
                self._write_local(record)

    def generation(
        self,
        name: str,
        *,
        model: str | None,
        system_prompt: str,
        user_prompt: str,
        model_parameters: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        prompt: Any = None,
    ):
        """LLM call observation; input is the role-labelled message list."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.observe(
            name,
            as_type="generation",
            input=messages,
            metadata=metadata,
            model=model,
            model_parameters=dict(model_parameters or {}),
            prompt=prompt,
        )

    def get_prompt(self, name: str) -> Any:
        """Read a cached production chat prompt; never publish during mining."""
        if self._client is None or not self._prompt_management or name in self._prompt_failures:
            return None
        try:
            return self._client.get_prompt(
                name, type="chat", label="production", max_retries=0, fetch_timeout_seconds=3,
            )
        except Exception as exc:
            self._prompt_failures.add(name)
            logger.warning("Using local prompt for %s (%s)", name, type(exc).__name__)
            return None

    def score_iteration(self, name: str, value: float, *, comment: str | None = None) -> None:
        """Attach a numeric score to the current iteration's trace."""
        if self._client is None or self._trace_id is None:
            return
        try:
            self._client.create_score(
                name=name,
                value=float(value),
                trace_id=self._trace_id,
                data_type="NUMERIC",
                comment=comment,
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("Langfuse score failed: %s", exc)

    def flush(self) -> None:
        if self._client is None:
            return
        try:
            self._client.flush()
        except Exception as exc:  # pragma: no cover
            logger.warning("Langfuse flush failed: %s", exc)

    # ------------------------------------------------------------------
    # Local mirror
    # ------------------------------------------------------------------

    def _write_local(self, record: dict[str, Any]) -> None:
        if self._local_path is None:
            return
        record = dict(record)
        # Store each distinct system prompt once; the log keeps its hash.
        messages = record.get("input")
        if record.get("type") == "generation" and isinstance(messages, list) and messages:
            first = messages[0]
            if isinstance(first, dict) and first.get("role") == "system":
                record["input"] = [
                    {"role": "system", "content_ref": self._store_system_prompt(str(first["content"]))},
                    *messages[1:],
                ]
        try:
            with open(self._local_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
        except OSError as exc:  # pragma: no cover
            logger.debug("Local trace write failed: %s", exc)

    def _store_system_prompt(self, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        if self._prompt_dir is not None:
            path = self._prompt_dir / f"{digest}.txt"
            if not path.exists():
                self._prompt_dir.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
        return f"system_prompts/{digest}.txt"


# ---------------------------------------------------------------------------
# Active tracer (one per process; the loop installs it for the run)
# ---------------------------------------------------------------------------


class _NullTracer:
    langfuse_enabled = False

    def get_prompt(self, name: str) -> Any:
        return None

    @contextmanager
    def iteration(self, iteration: int, *, input: Any = None) -> Iterator[Any]:
        yield _NullObservation()

    @contextmanager
    def observe(self, name: str, **_: Any) -> Iterator[Any]:
        yield _NullObservation()

    def generation(self, name: str, **_: Any):
        return self.observe(name)

    def score_iteration(self, name: str, value: float, *, comment: str | None = None) -> None:
        return None

    def flush(self) -> None:
        return None


_NULL_TRACER = _NullTracer()
_active: MiningTracer | None = None


def get_tracer() -> MiningTracer | _NullTracer:
    """Return the tracer installed for the current run, or a no-op tracer."""
    return _active if _active is not None else _NULL_TRACER


@contextmanager
def activate(tracer: MiningTracer | None) -> Iterator[None]:
    """Install ``tracer`` for the duration of a run, then flush it."""
    global _active
    previous = _active
    _active = tracer
    try:
        yield
    finally:
        _active = previous
        if tracer is not None:
            tracer.flush()


def tracer_from_config(
    config: Any,
    *,
    session_id: str,
    output_dir: str | Path | None,
    tags: list[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> MiningTracer:
    """Build a tracer from the ``observability`` config section (flat configs use defaults)."""
    section = getattr(config, "observability", None)
    return MiningTracer(
        session_id=session_id,
        output_dir=output_dir,
        langfuse=str(getattr(section, "langfuse", "auto")),
        local_log=bool(getattr(section, "local_log", True)),
        environment=str(getattr(section, "environment", "development")),
        release=getattr(section, "release", None) or None,
        sample_rate=float(getattr(section, "sample_rate", 1.0)),
        prompt_management=bool(getattr(section, "prompt_management", True)),
        tags=tags,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _ascii(text: str) -> str:
    return str(text).encode("ascii", "replace").decode("ascii")


def _short(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    # v5 JSON-encodes strings too. Quotes, backslashes and control characters
    # need more space than their raw length; otherwise the SDK drops the value.
    text = text[:_MAX_METADATA_VALUE - 2]
    while len(json.dumps(text, ensure_ascii=False)) > _MAX_METADATA_VALUE:
        text = text[:-1]
    return text


def _json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:  # pragma: no cover
        pass
    return str(value)
