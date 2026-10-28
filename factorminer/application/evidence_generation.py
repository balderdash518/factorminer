"""Evidence-guided proposal, deterministic checks, advisory review and re-ask.

The shared Ralph/Helix orchestration crosses this seam before numerical
evaluation. No rule here changes admission thresholds or reads holdout data.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any
from uuid import uuid4

from factorminer.agent.evidence_reviewer import EvidenceReviewer
from factorminer.agent.output_parser import candidate_pairs
from factorminer.memory.attempt_archive import (
    ARCHIVE_FILENAME,
    AttemptArchive,
    clean_proposal_metadata,
    formula_descriptor,
    json_safe,
    library_identity,
)
from factorminer.utils.tracing import get_tracer


@dataclass(frozen=True)
class EvidenceGuidedSettings:
    enabled: bool = False
    reserve_candidates: int = 2
    max_replacement_rounds: int = 2
    history_limit: int = 6
    match_limit: int = 3
    max_context_chars: int = 8000
    similarity_threshold: float = 0.65
    review_enabled: bool = True
    max_review_calls: int = 2
    max_review_candidates: int = 8
    review_max_tokens: int = 2048
    enable_embeddings: bool = False

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> EvidenceGuidedSettings:
        if raw is not None and not isinstance(raw, Mapping):
            raise ValueError("memory.evidence_guided must be a mapping")
        raw = dict(raw or {})
        unknown = raw.keys() - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown memory.evidence_guided keys: {sorted(unknown)}")
        settings = cls(**raw)
        for name in ("enabled", "review_enabled", "enable_embeddings"):
            if not isinstance(getattr(settings, name), bool):
                raise ValueError(f"memory.evidence_guided.{name} must be boolean")
        for name in (
            "reserve_candidates", "max_replacement_rounds", "history_limit", "match_limit",
            "max_context_chars", "max_review_calls", "max_review_candidates", "review_max_tokens",
        ):
            value = getattr(settings, name)
            minimum = 0 if name in {"reserve_candidates", "max_replacement_rounds", "max_review_calls"} else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"memory.evidence_guided.{name} must be an integer >= {minimum}")
        if settings.max_context_chars < 2000:
            raise ValueError("memory.evidence_guided.max_context_chars must be >= 2000")
        if not isinstance(settings.similarity_threshold, (int, float)) or not 0 < settings.similarity_threshold <= 1:
            raise ValueError("memory.evidence_guided.similarity_threshold must be in (0, 1]")
        return settings


def _book_usage(budget: Any, usage: Mapping[str, int]) -> None:
    calls = int(usage.get("calls", 0))
    if calls:
        budget.record_llm_call(int(usage.get("input", 0)), int(usage.get("output", 0)))
        budget.llm_calls += calls - 1


def _dump(value: Any) -> str:
    return json.dumps(json_safe(value), ensure_ascii=False, allow_nan=False)


class EvidenceGuidedGeneration:
    """Bounded history context and a checked proposal-to-observation lifecycle."""

    def __init__(self, settings: EvidenceGuidedSettings, output_dir: str, scope_id: str):
        self.settings = settings
        self.output_dir = output_dir
        self.base_scope = scope_id
        self.run_id = uuid4().hex
        self.archive = self._archive()
        self.pending: dict[str, str] = {}
        self.last_stats: dict[str, int] = {}
        self.last_feedback: dict[str, Any] = {}
        self._library_snapshot = ""
        self._last_iteration = 0

    def _archive(self, *, load_existing: bool = True) -> AttemptArchive:
        embedder = None
        if self.settings.enable_embeddings:
            from factorminer.memory.embeddings import FormulaEmbedder

            embedder = FormulaEmbedder()
        return AttemptArchive(self.output_dir, f"{self.base_scope}:{self.run_id}", embedder=embedder, load_existing=load_existing)

    def prepare(
        self, memory_signal: dict[str, Any], library_state: dict[str, Any], *,
        iteration: int, last_feedback: dict[str, Any], family_ledger: dict[str, Any],
        research_action: dict[str, Any],
    ) -> dict[str, Any]:
        goal = research_action.get("chosen") or memory_signal.get("research_cycle_theme") or {
            "recommended": memory_signal.get("recommended_directions", [])[:4],
            "hypotheses": last_feedback.get("reflection", {}).get("hypotheses", []),
        }
        query = _dump(goal)
        with get_tracer().observe("retrieve-attempt-history", as_type="retriever", input={"query": query}) as trace:
            hits = self.archive.retrieve(query, limit=self.settings.history_limit, diverse=True)
            # Render complete evidence rows, not a truncated JSON blob that loses
            # IDs or results. Trace only the records actually shown to generation.
            allowance = self.settings.max_context_chars
            blocks = ["ARCHIVE SUMMARY: " + _dump(self.archive.summary()), "RESEARCH GOAL: " + query[:allowance // 12]]
            blocks.append("PREVIOUS GENERATION FEEDBACK: " + _dump(self.last_feedback)[:allowance // 12])
            blocks.append("LAST BATCH FEEDBACK: " + _dump(last_feedback)[:allowance // 6])
            blocks.append("FAMILY HISTORY: " + _dump(family_ledger)[:allowance // 12])
            used = sum(len(b) + 1 for b in blocks)
            selected = []
            for hit in hits:
                block = "HISTORICAL EVIDENCE: " + _dump(hit)
                if used + len(block) + 1 > self.settings.max_context_chars:
                    continue
                selected.append(hit)
                blocks.append(block)
                used += len(block) + 1
            text = "\n".join(blocks)
            context = {
                **memory_signal, "evidence_guided": True, "attempt_history": selected,
                "attempt_history_text": text,
            }
            self.archive.event(
                "context_selected", iteration, query=query, selected_ids=[h["attempt_id"] for h in selected],
                selection_rule="hybrid_rank_with_success_failure_contrast", prompt_text=text,
            )
            trace.update(output={"selected": selected, "prompt_text": text})
        return context

    def _record_parser_exclusions(self, generator: Any, returned: list[tuple[str, str]], iteration: int) -> Counter[str]:
        counts: Counter[str] = Counter()
        lines = getattr(generator, "last_unparsed_lines", [])
        if lines:
            self.archive.event("unparsed_generation_output", iteration, lines=list(lines), measured=False)
            counts["generation_unparsed_lines"] += len(lines)
        remaining = Counter(returned)
        for proposal in getattr(generator, "last_proposals", []):
            pair = (proposal.name, proposal.formula)
            if remaining[pair] and proposal.is_valid:
                remaining[pair] -= 1
                continue
            attempt_id = self.archive.propose(
                *pair, iteration, metadata=clean_proposal_metadata(proposal.evidence), source="generator_parser",
            )
            self.archive.event(
                "invalid" if not proposal.is_valid else "parser_duplicate", iteration,
                attempt_id=attempt_id, rejection_reason=proposal.parse_error, measured=False,
                outcome="parse_error" if not proposal.is_valid else None,
            )
            counts["generation_proposals"] += 1
            counts["generation_invalid" if not proposal.is_valid else "generation_parser_duplicates"] += 1
        return counts

    def filter_batch(
        self, candidates: list[tuple[str, str]], *, generator: Any, budget: Any,
        memory_signal: dict[str, Any], library_state: dict[str, Any], iteration: int,
        target_size: int, allow_replacement: bool = True,
        source: str = "generation",
    ) -> tuple[list[tuple[str, str]], dict[str, int]]:
        self.pending = {}
        self._last_iteration = iteration
        self._library_snapshot = library_identity(library_state)
        stats: Counter[str] = Counter()
        accepted: list[tuple[str, str]] = []
        seen: dict[str, tuple[str, str]] = {}
        library_keys = {
            d["key"] for m in library_state.get("members", [])
            if (d := formula_descriptor(str(m.get("formula", "")))) is not None
        }
        reviewer = EvidenceReviewer(generator.llm_provider, max_tokens=self.settings.review_max_tokens)
        current = list(candidates)
        context = dict(memory_signal)
        notices: list[dict[str, Any]] = []
        if source == "generation":
            stats.update(self._record_parser_exclusions(generator, current, iteration))

        def approve(attempt_id: str, name: str, formula: str, key: str) -> None:
            if len(accepted) >= target_size:
                self.archive.event("reserve_unused", iteration, attempt_id=attempt_id, measured=False)
                return
            names = {n for n, _ in accepted}
            original, suffix = name, 2
            while name in names:
                name, suffix = f"{original}_{suffix}", suffix + 1
            accepted.append((name, formula))
            self.pending[key] = attempt_id
            seen[key] = (attempt_id, "approved")
            self.archive.event("approved", iteration, attempt_id=attempt_id, name=name, measured=False)

        max_rounds = self.settings.max_replacement_rounds if allow_replacement else 0
        for round_no in range(max_rounds + 1):
            metadata = {
                (p.name, p.formula): clean_proposal_metadata(p.evidence)
                for p in getattr(generator, "last_proposals", []) if p.is_valid and source == "generation"
            }
            requests = []
            for name, formula in current:
                stats["generation_proposals"] += 1
                meta = metadata.get((name, formula), {})
                attempt_id = self.archive.propose(name, formula, iteration, metadata=meta, source=source, round=round_no)
                descriptor = formula_descriptor(formula)
                error = ""
                if descriptor is None:
                    error = "Formula does not parse in the executor DSL"
                elif set(descriptor["features"]) - set(library_state.get("available_features", descriptor["features"])):
                    error = "Formula uses features unavailable in this dataset"
                if error:
                    self.archive.event("invalid", iteration, attempt_id=attempt_id, rejection_reason=error, measured=False)
                    stats["generation_invalid"] += 1
                    notices.append({"attempt_id": attempt_id, "formula": formula, "rule": "invalid", "reason": error})
                    continue
                key = descriptor["key"]
                prior = self.archive.exact_observation(key, self._library_snapshot)
                previous = seen.get(key)
                retention = (
                    previous and previous[1] == "revision_requested"
                    and meta.get("retention_reason") and meta.get("falsification_condition")
                )
                if key in library_keys or prior or (previous and not retention):
                    duplicate_id = prior["attempt_id"] if prior else (previous[0] if previous else "")
                    evidence = self.archive.attempts.get(duplicate_id, {})
                    reason = "Already in current library" if key in library_keys else "Already tested or proposed in this batch"
                    notice = {
                        "attempt_id": attempt_id, "formula": formula, "rule": "already_tested",
                        "duplicate_of": duplicate_id, "reason": reason,
                        "historical_outcome": evidence.get("outcome"), "historical_metrics": evidence.get("metrics", {}),
                    }
                    self.archive.event("already_tested", iteration, attempt_id=attempt_id, **{k: v for k, v in notice.items() if k != "attempt_id"}, measured=False)
                    stats["generation_duplicates"] += 1
                    notices.append(notice)
                    continue
                seen[key] = (attempt_id, "proposed")
                matches = self.archive.retrieve(
                    formula + " " + meta.get("hypothesis", ""), formula=formula,
                    limit=self.settings.match_limit, exclude_id=attempt_id,
                )
                declared = meta.get("declared_parent_id", "")
                if declared and declared not in self.archive.attempts:
                    self.archive.event("invalid_parent_reference", iteration, attempt_id=attempt_id, unverified_parent_id=declared)
                    declared = ""
                invalid_refs = [v for v in meta.get("reference_ids", []) if v not in self.archive.attempts]
                if invalid_refs:
                    self.archive.event("invalid_evidence_reference", iteration, attempt_id=attempt_id, unverified_reference_ids=invalid_refs)
                self.archive.event(
                    "history_matched", iteration, attempt_id=attempt_id,
                    nearest_prior_ids=[h["attempt_id"] for h in matches], matches=matches,
                    declared_parent_id=declared,
                )
                if retention:
                    self.archive.event("retained_with_reason", iteration, attempt_id=attempt_id, retention_reason=meta["retention_reason"])
                    approve(attempt_id, name, formula, key)
                    continue
                similar = [h for h in matches if h["similarity"]["structural"] >= self.settings.similarity_threshold or h["similarity"]["dense"] >= 0.85]
                if similar:
                    stats["generation_similar"] += 1
                request = {"attempt_id": attempt_id, "name": name, "formula": formula,
                           "metadata": meta, "history": similar, "formula_key": key}
                review_fits = len(_dump([*requests, request])) + 500 <= self.settings.max_context_chars
                if (
                    similar and self.settings.review_enabled
                    and source != "refinement"
                    and stats["generation_review_calls"] < self.settings.max_review_calls
                    and len(requests) < self.settings.max_review_candidates
                    and len(accepted) < target_size
                    and review_fits
                ):
                    requests.append(request)
                else:
                    if similar:
                        self.archive.event("similarity_warning", iteration, attempt_id=attempt_id, reason="Advisory review disabled or capped; similarity alone is not a veto")
                    approve(attempt_id, name, formula, key)

            if requests:
                with get_tracer().observe("review-candidate-experiments", as_type="agent", input={"candidates": requests}) as trace:
                    reviews = reviewer.review(requests, {
                        "available_features": library_state.get("available_features", []),
                        "target_size": target_size, "scope": self.run_id,
                    })
                    _book_usage(budget, reviewer.last_usage)
                    stats["generation_review_calls"] += 1
                    trace.update(output={"reviews": reviews, "abstained": len(requests) - len(reviews)})
                for request in requests:
                    attempt_id, key = request["attempt_id"], request["formula_key"]
                    review = reviews.get(attempt_id, {"action": "keep", "status": "abstained", "reason": "Review unavailable, malformed, or unsupported by supplied IDs"})
                    self.archive.event("reviewed", iteration, attempt_id=attempt_id, review=review)
                    if review["action"] == "keep":
                        approve(attempt_id, request["name"], request["formula"], key)
                    else:
                        seen[key] = (attempt_id, "revision_requested")
                        stats["generation_revision_requests"] += 1
                        self.archive.event("revision_requested", iteration, attempt_id=attempt_id, measured=False)
                        notices.append({**request, "rule": "reviewer_advice", "review": review})

            if len(accepted) >= target_size or round_no >= max_rounds:
                break
            # A bounded replacement call is a new proposal, never an evaluation
            # shortcut. It goes through exactly the same checks on the next pass.
            notice_payload: dict[str, Any] = {
                "needed": target_size - len(accepted),
                "accepted_formulas": [], "notices": [],
                "instruction": "Replace deterministic rejects. For reviewer advice, revise or explain a purposeful controlled test using retention_reason and falsification_condition.",
            }
            # Large batches must not make the replacement context unbounded.
            # Full accepted-set checking remains deterministic even when only
            # some accepted formulas fit into the LLM's context allowance.
            for _, formula in accepted:
                trial = {**notice_payload, "accepted_formulas": [*notice_payload["accepted_formulas"], formula]}
                if len(_dump(trial)) <= self.settings.max_context_chars // 2:
                    notice_payload = trial
            for notice in reversed(notices):
                trial = {**notice_payload, "notices": [*notice_payload["notices"], notice]}
                if len(_dump(trial)) <= self.settings.max_context_chars:
                    notice_payload = trial
            context = {**memory_signal, "replacement_notice": _dump(notice_payload)}
            self.archive.event("replacement_requested", iteration, round=round_no + 1, context=notice_payload)
            try:
                proposals = generator.generate_batch(
                    memory_signal=context, library_state=library_state,
                    batch_size=target_size - len(accepted) + self.settings.reserve_candidates,
                )
            finally:
                _book_usage(budget, getattr(generator, "last_batch_usage", {}))
            current = candidate_pairs(proposals)
            stats.update(self._record_parser_exclusions(generator, current, iteration))
            stats["generation_replacement_rounds"] += 1

        stats["generation_approved"] = len(accepted)
        stats["generation_shortfall"] = max(0, target_size - len(accepted))
        for name in (
            "generation_proposals", "generation_invalid", "generation_duplicates",
            "generation_parser_duplicates", "generation_unparsed_lines", "generation_similar",
            "generation_review_calls", "generation_revision_requests", "generation_replacement_rounds",
        ):
            stats.setdefault(name, 0)
        self.last_stats = dict(stats)
        self.last_feedback = {"iteration": iteration, **dict(stats), "note": "Rule rejections and LLM opinions are unmeasured, not low-IC outcomes."}
        self.archive.event("generation_feedback", iteration, feedback=self.last_feedback)
        return accepted, dict(stats)

    def record_results(self, results: list[Any], *, iteration: int) -> None:
        """Called after final admission (including Helix checks), not before it."""
        for result in results:
            descriptor = formula_descriptor(result.formula)
            attempt_id = self.pending.get(descriptor["key"]) if descriptor else None
            if not attempt_id:
                attempt_id = self.archive.propose(result.factor_name, result.formula, iteration, source="evaluation_dispatch")
            outcome = str(getattr(result, "outcome", "") or "unknown")
            measured = bool(result.parse_ok) and outcome not in {"parse_error", "compute_error", "invalid_signal", "unknown"}
            result.attempt_id = attempt_id
            record = self.archive.attempts[attempt_id]
            result.declared_parent_id = record.get("declared_parent_id", "")
            result.nearest_prior_ids = record.get("nearest_prior_ids", [])
            self.archive.event(
                "evaluated", iteration, attempt_id=attempt_id, measured=measured,
                outcome=outcome, admitted=bool(result.admitted), library_snapshot=self._library_snapshot,
                rejection_reason=result.rejection_reason,
                metrics={k: getattr(result, k, None) if measured else None for k in (
                    "ic_mean", "ic_paper_mean", "icir", "ic_paper_icir", "max_correlation", "stage_passed", "portfolio",
                )},
                correlated_with=getattr(result, "correlated_with", ""),
            )

    def save_checkpoint(self, checkpoint_dir: Path) -> dict[str, str]:
        self.archive.save_snapshot(checkpoint_dir / ARCHIVE_FILENAME)
        return {"base_scope": self.base_scope, "run_id": self.run_id}

    def restore_checkpoint(self, checkpoint_dir: Path, state: Mapping[str, Any]) -> None:
        if state.get("base_scope") != self.base_scope:
            raise ValueError("Evidence-guided checkpoint scope changed; use the original mining configuration")
        self.run_id = str(state["run_id"])
        if not (checkpoint_dir / ARCHIVE_FILENAME).exists():
            raise ValueError("Evidence-guided checkpoint is missing its attempt archive")
        self.archive = self._archive(load_existing=False)
        self.archive.import_events(checkpoint_dir / ARCHIVE_FILENAME)
        self.last_feedback = dict(self.archive.last_generation_feedback)
