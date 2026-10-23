"""Reusable orchestration helpers for Ralph and Helix loops.

This module keeps the loop classes focused on policy decisions while
centralizing the repeated stage-chain execution and iteration telemetry.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from factorminer.architecture import IterationPayload
from factorminer.utils.logging import FactorRecord, IterationRecord
from factorminer.utils.tracing import get_tracer

logger = logging.getLogger(__name__)

# Stage -> (observation name, Langfuse observation type). Names are part of the
# trace contract (dashboards and filters key on them); keep them stable.
_STAGE_OBSERVATIONS = {
    "retrieve": ("retrieve-memory", "retriever"),
    "generate": ("generate-candidates", "chain"),
    "evaluate": ("evaluate-candidates", "evaluator"),
    "library_update": ("update-library", "span"),
    "distill": ("distill-memory", "span"),
}


@dataclass(slots=True)
class IterationTelemetry:
    """Structured snapshot of one completed iteration."""

    iteration: int
    candidates_generated: int
    stats: dict[str, Any]
    elapsed_seconds: float
    results: Sequence[Any] = field(default_factory=tuple)


class LoopExecutionService:
    """Shared orchestration helpers for factor-mining loops."""

    def __init__(self, loop: Any) -> None:
        self.loop = loop

    def new_payload(self, batch_size: int) -> IterationPayload:
        return IterationPayload(iteration=self.loop.iteration, batch_size=batch_size)

    def run_stage_chain(
        self,
        payload: IterationPayload,
        stage_names: Sequence[str],
    ) -> None:
        profile = getattr(self.loop, "runtime_profile", None)
        tracer = get_tracer()
        for stage_name in stage_names:
            name, as_type = _STAGE_OBSERVATIONS.get(
                stage_name, (stage_name.replace("_", "-"), "span")
            )
            with tracer.observe(
                name, as_type=as_type, input=self._stage_input(stage_name, payload)
            ) as observation:
                if profile is None:
                    self.loop.stages[stage_name].run(self.loop, payload)
                else:
                    with profile.stage(stage_name):
                        self.loop.stages[stage_name].run(self.loop, payload)
                    if stage_name == "evaluate":
                        profile.record_panels(
                            stage_name,
                            (getattr(result, "signals", None) for result in payload.results),
                        )
                        profile.record_outcomes(stage_name, payload.results)
                observation.update(output=self._safe_stage_output(stage_name, payload))

    # ------------------------------------------------------------------
    # Trace payloads: what a reviewer needs to see for each stage
    # ------------------------------------------------------------------

    def _stage_input(self, stage_name: str, payload: IterationPayload) -> dict[str, Any]:
        if stage_name == "retrieve":
            return {"library_size": self.loop.library.size}
        if stage_name == "generate":
            return {"batch_size": payload.batch_size}
        if stage_name == "evaluate":
            return {"candidates": len(payload.candidates)}
        if stage_name == "library_update":
            return {"passed_evaluation": sum(1 for r in payload.results if getattr(r, "admitted", False))}
        return {}

    def _safe_stage_output(self, stage_name: str, payload: IterationPayload) -> dict[str, Any]:
        # Trace summaries must never fail an iteration (custom stages/loops
        # may carry duck-typed payloads).
        try:
            return self._stage_output(stage_name, payload)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Trace summary for %s failed: %s", stage_name, exc)
            return {"summary_error": str(exc)}

    def _stage_output(self, stage_name: str, payload: IterationPayload) -> dict[str, Any]:
        if stage_name == "retrieve":
            signal = payload.memory_signal
            return {
                "memory_prompt": str(signal.get("prompt_text", "") or ""),
                "recommended": len(signal.get("recommended_directions", []) or []),
                "forbidden": len(signal.get("forbidden_directions", []) or []),
                "insights": len(signal.get("insights", []) or []),
                "library_state": payload.library_state.get("categories", {}),
            }
        if stage_name == "generate":
            generator = getattr(self.loop, "generator", None)
            return {
                "candidates": [{"name": n, "formula": f} for n, f in payload.candidates],
                "llm_usage": dict(getattr(generator, "last_batch_usage", {}) or {}),
            }
        if stage_name == "evaluate":
            return {
                "outcomes": dict(Counter(_outcome(r) for r in payload.results)),
                "candidates": [_result_row(r) for r in payload.results],
            }
        if stage_name == "library_update":
            return {
                "admitted": [
                    {"name": getattr(r, "factor_name", ""), "formula": getattr(r, "formula", ""),
                     "paper_ic": getattr(r, "ic_paper_mean", None),
                     "replaced": getattr(r, "replaced", None)}
                    for r in payload.admitted_results
                ],
                "library_size": self.loop.library.size,
            }
        if stage_name == "distill":
            memory = getattr(self.loop, "memory", None)
            if memory is None:
                return {}
            return {
                "memory_version": memory.version,
                "batch_feedback": dict(memory.state.last_batch_feedback),
                "success_patterns": [
                    {"name": p.name, "count": p.occurrence_count}
                    for p in memory.success_patterns[:20]
                ],
                "forbidden_directions": [
                    {"name": f.name, "count": f.occurrence_count,
                     "rho": round(float(f.typical_correlation), 3)}
                    for f in memory.forbidden_directions[:20]
                ],
                "insights": [i.insight for i in memory.insights[:10]],
            }
        return {}

    def execute_iteration(
        self,
        batch_size: int,
        *,
        trailing_stages: Sequence[str] = (),
        phase2_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute an iteration, retaining failed action accounting for resume."""
        tracer = get_tracer()
        with tracer.iteration(
            self.loop.iteration,
            input={
                "iteration": self.loop.iteration,
                "library_size": self.loop.library.size,
                "batch_size": batch_size,
            },
        ) as root:
            try:
                stats = self._execute_iteration(
                    batch_size, trailing_stages=trailing_stages, phase2_summary=phase2_summary,
                )
            except BaseException as exc:
                actions = getattr(self.loop, "research_actions", None)
                if actions is not None:
                    actions.fail(exc)
                raise
            self._trace_iteration_result(root, stats)
            return stats

    def _trace_iteration_result(self, root: Any, stats: dict[str, Any]) -> None:
        """Iteration output plus scores used to find where candidates are lost."""
        try:
            self._trace_iteration_scores(root, stats)
        except Exception as exc:  # noqa: BLE001 - tracing never fails an iteration
            logger.debug("Iteration trace summary failed: %s", exc)

    def _trace_iteration_scores(self, root: Any, stats: dict[str, Any]) -> None:
        results = list(getattr(self, "_last_results", ()) or ())
        outcomes = Counter(_outcome(r) for r in results)
        root.update(output={
            "admitted": [
                {"name": getattr(r, "factor_name", ""), "formula": getattr(r, "formula", ""),
                 "paper_ic": getattr(r, "ic_paper_mean", None)}
                for r in results if getattr(r, "admitted", False)
            ],
            "outcomes": dict(outcomes),
            "yield_rate": stats.get("yield_rate", 0.0),
            "library_size": stats.get("library_size", 0),
            "research_action": stats.get("research_action", {}).get("chosen", {}).get("kind")
            if isinstance(stats.get("research_action"), dict) else None,
        })
        tracer = get_tracer()
        n = max(len(results), 1)
        tracer.score_iteration("yield_rate", float(stats.get("yield_rate", 0.0)))
        tracer.score_iteration("library_size", float(stats.get("library_size", 0)))
        if results:
            tracer.score_iteration(
                "redundancy_rate",
                (outcomes["redundant"] + outcomes["batch_duplicate"]) / n,
            )
            tracer.score_iteration("weak_signal_rate", (outcomes["low_ic"] + outcomes["low_icir"]) / n)
            tracer.score_iteration(
                "implementation_failure_rate",
                (outcomes["parse_error"] + outcomes["compute_error"] + outcomes["invalid_signal"]) / n,
            )
            tracer.score_iteration(
                "best_paper_ic", max(float(getattr(r, "ic_paper_mean", 0.0) or 0.0) for r in results)
            )

    def _execute_iteration(
        self,
        batch_size: int,
        *,
        trailing_stages: Sequence[str] = (),
        phase2_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute the canonical mining sequence for a loop composition."""
        started_at = time.time()
        self._last_results = ()
        payload = self.new_payload(batch_size)
        self.run_stage_chain(payload, ("retrieve",))
        retrieval = self.loop.research_knowledge.retrieve(
            payload.library_state,
            limit=self.loop.settings.research_knowledge_retrieval_limit,
        )
        payload.memory_signal = retrieval.enrich(payload.memory_signal)
        actions = getattr(self.loop, "research_actions", None)
        action_kind = actions.prepare(payload) if actions is not None else "generate"
        if actions is not None and action_kind in ("stop", "challenge"):
            if action_kind == "challenge":
                actions.challenge(payload)
            elapsed = time.time() - started_at
            self.loop.budget.record_compute(elapsed)
            actions.finish(payload, elapsed)
            stats = self.build_stats(payload, elapsed)
            self.log_telemetry(self.build_telemetry(payload, stats, elapsed))
            return stats
        if actions is not None and action_kind == "refine":
            with get_tracer().observe(
                "refine-candidates", as_type="tool", input=dict(payload.research_action)
            ) as observation:
                actions.refine(payload)
                observation.update(output={
                    "candidates": [{"name": n, "formula": f} for n, f in payload.candidates],
                })
        else:
            self.run_stage_chain(payload, ("generate",))
            self._record_llm_usage()

        if not payload.candidates:
            logger.warning(
                "%s iteration %d: %s",
                self.loop._loop_type().title(),
                self.loop.iteration,
                self.describe_empty_generation(payload),
            )
            if actions is None:
                return self.empty_stats()
            elapsed = time.time() - started_at
            self.loop.budget.record_compute(elapsed)
            actions.finish(payload, elapsed)
            stats = self.build_stats(payload, elapsed)
            self.log_telemetry(self.build_telemetry(payload, stats, elapsed))
            return stats

        if actions is not None:
            actions.record_evaluation_dispatch(payload)
        self.run_stage_chain(payload, ("evaluate", "library_update"))
        summary = dict(phase2_summary or {})
        if "phase2_rejections" in payload.stage_metrics:
            summary["phase2_rejections"] = payload.stage_metrics["phase2_rejections"]
        self.loop._attach_factor_provenance(
            payload.admitted_results,
            library_state={
                **payload.library_state,
                "diagnostics": self.loop.library.get_diagnostics(),
            },
            memory_signal=payload.memory_signal,
            phase2_summary=summary,
            generator_family=("deterministic_refinement" if action_kind == "refine"
                              else self.loop._generator_family()),
        )

        self.run_stage_chain(payload, ("distill", *trailing_stages))
        self.loop.research_knowledge.record_results(
            payload.results,
            payload.memory_signal,
            iteration=self.loop.iteration,
        )
        elapsed = time.time() - started_at
        self.loop.budget.record_compute(elapsed)
        if actions is not None:
            actions.finish(payload, elapsed)
        stats = self.build_stats(payload, elapsed)
        self._last_results = tuple(payload.results)
        telemetry = self.build_telemetry(
            payload,
            stats,
            elapsed,
            candidates_generated=self.candidate_count(payload),
        )
        self.log_telemetry(telemetry)
        return stats

    def _record_llm_usage(self) -> None:
        """Book the generator's real call count and tokens (0 when unknown)."""
        usage = dict(getattr(getattr(self.loop, "generator", None), "last_batch_usage", {}) or {})
        self.loop.budget.record_llm_call(
            prompt_tokens=int(usage.get("input", 0)),
            completion_tokens=int(usage.get("output", 0)),
        )
        extra_calls = int(usage.get("calls", 1)) - 1
        if extra_calls > 0:
            self.loop.budget.llm_calls += extra_calls

    def candidate_count(self, payload: IterationPayload) -> int:
        if "candidates_before_canon" in payload.stage_metrics:
            return int(payload.stage_metrics["candidates_before_canon"])
        return len(payload.candidates)

    def empty_stats(self) -> dict[str, Any]:
        return self.loop._empty_stats()

    def build_stats(self, payload: IterationPayload, elapsed: float) -> dict[str, Any]:
        stats = self.loop._compute_stats(payload.results, payload.admitted_results, elapsed)
        stats.update(payload.stage_metrics)
        return stats

    def describe_empty_generation(self, payload: IterationPayload) -> str:
        if self.candidate_count(payload) > 0:
            return "all candidates removed by canonicalization"
        return "generator produced 0 candidates"

    def zero_admission_guidance(self, *, target_size: int, max_iterations: int) -> str | None:
        """Explain likely causes when a run produced candidates but admitted nothing."""
        session = getattr(self.loop, "_session", None)
        if session is None:
            return None

        summary = session.get_summary()
        total_candidates = int(summary.get("total_candidates", 0) or 0)
        total_admitted = int(summary.get("total_admitted", 0) or 0)
        if total_candidates <= 0 or total_admitted > 0 or self.loop.library.size > 0:
            return None

        settings = getattr(self.loop, "settings", getattr(self.loop, "config", None))
        data_tensor = getattr(self.loop, "data_tensor", None)
        if data_tensor is not None and getattr(data_tensor, "ndim", 0) >= 2:
            panel = f"{data_tensor.shape[0]} assets x {data_tensor.shape[1]} periods"
        else:
            panel = "unknown panel size"

        ic_threshold = getattr(settings, "ic_threshold", "unknown")
        icir_threshold = getattr(settings, "icir_threshold", "unknown")
        corr_threshold = getattr(settings, "correlation_threshold", "unknown")

        return (
            "No factors were admitted after "
            f"{session.total_iterations} iterations and {total_candidates} evaluated candidates. "
            "This can be normal for smoke tests, tiny samples, strict IC/ICIR thresholds, or "
            "high redundancy pressure. "
            f"Panel={panel}; target={target_size}; max_iterations={max_iterations}; "
            f"thresholds: ic={ic_threshold}, icir={icir_threshold}, "
            f"correlation={corr_threshold}. "
            "For exploration, try more iterations/bigger batches, a larger time panel, lower "
            "screening thresholds, or `--mock` first to isolate setup from data quality."
        )

    def build_telemetry(
        self,
        payload: IterationPayload,
        stats: dict[str, Any],
        elapsed: float,
        *,
        candidates_generated: int | None = None,
    ) -> IterationTelemetry:
        return IterationTelemetry(
            iteration=self.loop.iteration,
            candidates_generated=candidates_generated
            if candidates_generated is not None
            else self.candidate_count(payload),
            stats=stats,
            elapsed_seconds=elapsed,
            results=tuple(payload.results),
        )

    def log_telemetry(self, telemetry: IterationTelemetry) -> None:
        """Emit the shared iteration telemetry to reporter and session logs."""
        self.loop.reporter.log_batch(**telemetry.stats)
        session_logger = getattr(self.loop, "_session_logger", None)
        if session_logger is None:
            return

        ic_values = [
            getattr(r, "ic_paper_mean", r.ic_mean)
            for r in telemetry.results
            if getattr(r, "parse_ok", False)
        ]
        record = IterationRecord(
            iteration=telemetry.iteration,
            candidates_generated=telemetry.candidates_generated,
            ic_passed=int(telemetry.stats.get("ic_passed", 0)),
            correlation_passed=int(telemetry.stats.get("corr_passed", 0)),
            admitted=int(telemetry.stats.get("admitted", 0)),
            rejected=max(telemetry.candidates_generated - int(telemetry.stats.get("admitted", 0)), 0),
            replaced=int(telemetry.stats.get("replaced", 0)),
            library_size=int(telemetry.stats.get("library_size", 0)),
            best_ic=max(ic_values) if ic_values else 0.0,
            mean_ic=float(np.mean(ic_values)) if ic_values else 0.0,
            elapsed_seconds=telemetry.elapsed_seconds,
        )
        session_logger.log_iteration(record)

        for result in telemetry.results:
            factor_rec = FactorRecord(
                expression=result.formula,
                ic=getattr(result, "ic_paper_mean", result.ic_mean)
                if getattr(result, "parse_ok", False)
                else None,
                icir=getattr(result, "ic_paper_icir", result.icir)
                if getattr(result, "parse_ok", False)
                else None,
                max_correlation=(
                    result.max_correlation if getattr(result, "parse_ok", False) else None
                ),
                admitted=bool(getattr(result, "admitted", False)),
                rejection_reason=result.rejection_reason or None,
                replaced_factor=str(result.replaced) if getattr(result, "replaced", None) else None,
            )
            session_logger.log_factor(factor_rec)


def _outcome(result: Any) -> str:
    return str(getattr(result, "outcome", "") or "unknown")


def _result_row(result: Any) -> dict[str, Any]:
    """One evaluated candidate as shown in the trace."""
    return {
        "name": result.factor_name,
        "formula": result.formula,
        "outcome": result.outcome,
        "paper_ic": round(float(result.ic_paper_mean or 0.0), 5),
        "icir": round(float(result.ic_paper_icir or 0.0), 4),
        "max_correlation": round(float(result.max_correlation or 0.0), 4),
        "correlated_with": result.correlated_with,
        "reason": result.rejection_reason,
    }
