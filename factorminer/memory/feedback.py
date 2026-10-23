"""Structured per-batch feedback distilled from a mining trajectory tau.

The paper's formation operator F compresses tau into direction-level memory
(P_succ / P_fail / I). That compression drops the concrete evidence the
generator needs to act on next: which strong candidates were redundant and
with which library factor, how close weak candidates came to the IC gate,
and which formulas did not run at all. This module keeps that evidence as a
small, deterministic record (every number comes from the evaluator, none
from the LLM) that is stored in the mining state and rendered into the next
prompt.

Implementation failures (parse/compute/invalid signal) are reported apart
from quality and redundancy failures: a formula that did not run says
nothing about the economic idea behind it.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

PARSE_ERROR = "parse_error"
COMPUTE_ERROR = "compute_error"
INVALID_SIGNAL = "invalid_signal"
LOW_IC = "low_ic"
LOW_ICIR = "low_icir"
REDUNDANT = "redundant"
BATCH_DUPLICATE = "batch_duplicate"
RESEARCH_REJECTED = "research_rejected"
ADMITTED = "admitted"
REPLACED = "replaced"
REJECTED = "rejected"  # legacy entry without enough detail to classify

IMPLEMENTATION_OUTCOMES = frozenset({PARSE_ERROR, COMPUTE_ERROR, INVALID_SIGNAL})
QUALITY_OUTCOMES = frozenset({LOW_IC, LOW_ICIR})
REDUNDANCY_OUTCOMES = frozenset({REDUNDANT, BATCH_DUPLICATE})
SUCCESS_OUTCOMES = frozenset({ADMITTED, REPLACED})

FEEDBACK_HISTORY_LIMIT = 10
_MAX_LISTED = 5


def candidate_outcome(entry: Mapping[str, Any]) -> str:
    """Return the structured outcome of one trajectory entry.

    Prefers the evaluator's ``outcome`` field; older trajectories without it
    are classified from the admission flag and rejection text.
    """
    outcome = str(entry.get("outcome", "") or "")
    if outcome:
        return outcome
    if entry.get("admitted", False):
        return REPLACED if entry.get("replaced") is not None else ADMITTED
    reason = str(entry.get("rejection_reason", "") or "").lower()
    if "parse" in reason:
        return PARSE_ERROR
    if "signal computation" in reason:
        return COMPUTE_ERROR
    if "nan" in reason or "constant signal" in reason:
        return INVALID_SIGNAL
    if "deduplication" in reason:
        return BATCH_DUPLICATE
    if "icir" in reason:
        return LOW_ICIR
    if "dependence" in reason or "correlation" in reason:
        return REDUNDANT
    if " ic " in f" {reason} " or "ic " in reason:
        return LOW_IC
    if float(entry.get("max_correlation", 0.0) or 0.0) >= 0.5:
        return REDUNDANT
    return REJECTED


def is_redundancy_failure(entry: Mapping[str, Any]) -> bool:
    return candidate_outcome(entry) in REDUNDANCY_OUTCOMES


def is_implementation_failure(entry: Mapping[str, Any]) -> bool:
    return candidate_outcome(entry) in IMPLEMENTATION_OUTCOMES


def _quality(entry: Mapping[str, Any]) -> float:
    for key in ("ic_paper_mean", "paper_ic"):
        value = entry.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    try:
        return abs(float(entry.get("ic", 0.0) or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _icir(entry: Mapping[str, Any]) -> float:
    for key in ("paper_icir", "icir"):
        value = entry.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


def _short(formula: str, limit: int = 160) -> str:
    formula = str(formula or "")
    return formula if len(formula) <= limit else formula[: limit - 3] + "..."


def summarize_trajectory(
    trajectory: Iterable[Mapping[str, Any]],
    iteration: int,
    *,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the structured feedback record for one evaluated batch."""
    entries = [dict(e) for e in trajectory]
    for entry in entries:
        entry["_outcome"] = candidate_outcome(entry)
    counts = Counter(entry["_outcome"] for entry in entries)
    total = len(entries)

    admitted = sorted(
        (e for e in entries if e["_outcome"] in SUCCESS_OUTCOMES), key=_quality, reverse=True
    )
    redundant = sorted(
        (e for e in entries if e["_outcome"] in REDUNDANCY_OUTCOMES), key=_quality, reverse=True
    )
    weak = sorted(
        (e for e in entries if e["_outcome"] in QUALITY_OUTCOMES), key=_quality, reverse=True
    )
    broken = [e for e in entries if e["_outcome"] in IMPLEMENTATION_OUTCOMES]

    crowded = Counter(
        str(e.get("correlated_with", "") or "")
        for e in redundant
        if e["_outcome"] == REDUNDANT and e.get("correlated_with")
    )

    feedback: dict[str, Any] = {
        "iteration": int(iteration),
        "total": total,
        "counts": dict(counts),
        "yield_rate": round(len(admitted) / total, 4) if total else 0.0,
        "admitted": [
            {
                "name": str(e.get("factor_id", "")),
                "formula": _short(e.get("formula", "")),
                "ic": round(_quality(e), 4),
                "icir": round(_icir(e), 3),
                "max_correlation": round(float(e.get("max_correlation", 0.0) or 0.0), 3),
                "replaced": e.get("replaced"),
            }
            for e in admitted[:_MAX_LISTED]
        ],
        "near_misses": [
            {
                "name": str(e.get("factor_id", "")),
                "formula": _short(e.get("formula", "")),
                "ic": round(_quality(e), 4),
                "max_correlation": round(float(e.get("max_correlation", 0.0) or 0.0), 3),
                "correlated_with": str(e.get("correlated_with", "") or ""),
                "conflict_count": int(e.get("conflict_count", 0) or 0),
                "duplicate_of": str(e.get("duplicate_of", "") or ""),
                "note": str(e.get("replacement_note", "") or ""),
            }
            for e in redundant[:_MAX_LISTED]
        ],
        "weak": [
            {
                "name": str(e.get("factor_id", "")),
                "formula": _short(e.get("formula", "")),
                "ic": round(_quality(e), 4),
                "icir": round(_icir(e), 3),
                "reason": str(e.get("rejection_reason", "") or ""),
            }
            for e in weak[:3]
        ],
        "implementation_errors": [
            {
                "name": str(e.get("factor_id", "")),
                "formula": _short(e.get("formula", "")),
                "reason": _short(e.get("rejection_reason", ""), 200),
            }
            for e in broken[:_MAX_LISTED]
        ],
        "crowded_factors": [
            {"factor": name, "collisions": count} for name, count in crowded.most_common(3)
        ],
    }
    feedback["diagnosis"] = _diagnose(feedback, entries, history or [])
    return feedback


def _diagnose(
    feedback: Mapping[str, Any],
    entries: list[dict[str, Any]],
    history: list[dict[str, Any]],
) -> list[str]:
    """Deterministic, evidence-bound reading of the batch outcome."""
    total = int(feedback["total"])
    if total == 0:
        return []
    counts: Mapping[str, int] = feedback["counts"]
    n_broken = sum(counts.get(o, 0) for o in IMPLEMENTATION_OUTCOMES)
    n_weak = sum(counts.get(o, 0) for o in QUALITY_OUTCOMES)
    n_redundant = sum(counts.get(o, 0) for o in REDUNDANCY_OUTCOMES)
    n_success = sum(counts.get(o, 0) for o in SUCCESS_OUTCOMES)
    evaluated = max(total - n_broken, 1)
    notes: list[str] = []

    if n_broken / total >= 0.2:
        notes.append(
            f"{n_broken}/{total} formulas did not run (parse/compute/invalid). "
            "Fix syntax and operator arity first; these say nothing about the ideas."
        )
    if n_weak / evaluated >= 0.5:
        ics = sorted(_quality(e) for e in entries if e["_outcome"] in QUALITY_OUTCOMES)
        median = ics[len(ics) // 2] if ics else 0.0
        notes.append(
            f"{n_weak}/{evaluated} valid formulas were too weak (median |IC| {median:.4f}). "
            "More nesting or window tweaks rarely fix this; change the underlying signal."
        )
    if n_redundant / evaluated >= 0.3:
        crowded = feedback.get("crowded_factors") or []
        where = ", ".join(c["factor"] for c in crowded) if crowded else "the existing library"
        notes.append(
            f"{n_redundant}/{evaluated} valid formulas duplicated existing signals; "
            f"most collisions were with {where}. Move to a different mechanism or data field."
        )
    near = feedback.get("near_misses") or []
    multi = [m for m in near if m.get("conflict_count", 0) > 1]
    if near and len(multi) == len(near):
        notes.append(
            "Strong but redundant candidates each overlapped several library factors, "
            "so replacement was impossible; small variations will not escape that cluster."
        )
    single = [m for m in near if m.get("conflict_count", 0) == 1]
    if single:
        best = single[0]
        notes.append(
            f"{best['name']} (|IC| {best['ic']:.4f}) overlapped only {best['correlated_with']}; "
            f"a clearly stronger variant could replace it ({best['note'] or 'see replacement rule'})."
        )
    if n_success:
        notes.append(
            f"{n_success} admitted; their neighbourhoods are now occupied, so the next batch "
            "should complement rather than imitate them."
        )

    recent = [h for h in history[-2:]] + [{"yield_rate": feedback["yield_rate"]}]
    if len(recent) >= 3 and all(float(h.get("yield_rate", 0.0)) == 0.0 for h in recent):
        notes.append("Three consecutive batches admitted nothing: pivot to an unexplored mechanism.")
    return notes


def history_row(feedback: Mapping[str, Any]) -> dict[str, Any]:
    """Compact per-iteration row kept for trend reading."""
    return {
        "iteration": feedback.get("iteration", 0),
        "total": feedback.get("total", 0),
        "counts": dict(feedback.get("counts", {})),
        "yield_rate": feedback.get("yield_rate", 0.0),
    }


def render_batch_feedback(
    feedback: Mapping[str, Any] | None,
    history: list[Mapping[str, Any]] | None = None,
) -> str:
    """Render the feedback record as prompt text."""
    if not feedback or not feedback.get("total"):
        return ""
    counts: Mapping[str, int] = feedback.get("counts", {})
    order = [
        ADMITTED, REPLACED, REDUNDANT, BATCH_DUPLICATE, LOW_IC, LOW_ICIR,
        RESEARCH_REJECTED, PARSE_ERROR, COMPUTE_ERROR, INVALID_SIGNAL, REJECTED,
    ]
    count_text = ", ".join(f"{name}={counts[name]}" for name in order if counts.get(name))
    lines = [
        f"=== FEEDBACK FROM LAST BATCH (iteration {feedback.get('iteration', '?')}) ===",
        f"{feedback['total']} candidates: {count_text}",
    ]
    if history:
        trend = ", ".join(
            f"it{h.get('iteration', '?')}={float(h.get('yield_rate', 0.0)):.0%}"
            for h in list(history)[-5:]
        )
        lines.append(f"Admission yield trend: {trend}")
    for note in feedback.get("diagnosis", []):
        lines.append(f"- {note}")

    if feedback.get("admitted"):
        lines.append("Admitted:")
        for item in feedback["admitted"]:
            replaced = f", replaced #{item['replaced']}" if item.get("replaced") is not None else ""
            lines.append(
                f"  + {item['formula']}  (|IC| {item['ic']:.4f}, ICIR {item['icir']:.2f}, "
                f"max rho {item['max_correlation']:.2f}{replaced})"
            )
    if feedback.get("near_misses"):
        lines.append("Strong but redundant (do not resubmit near-copies):")
        for item in feedback["near_misses"]:
            if item.get("duplicate_of"):
                target = f"duplicate of batch member {item['duplicate_of']}"
            else:
                target = (
                    f"rho {item['max_correlation']:.2f} with {item['correlated_with'] or 'library'}"
                    f"; conflicts with {item['conflict_count']} factor(s)"
                )
            lines.append(f"  ~ {item['formula']}  (|IC| {item['ic']:.4f}; {target})")
    if feedback.get("weak"):
        lines.append("Closest to the IC gate but rejected:")
        for item in feedback["weak"]:
            lines.append(f"  - {item['formula']}  (|IC| {item['ic']:.4f}, ICIR {item['icir']:.2f})")
    if feedback.get("implementation_errors"):
        lines.append("Did not run (fix the formula, not the idea):")
        for item in feedback["implementation_errors"]:
            lines.append(f"  ! {item['formula']}  -> {item['reason']}")
    reflection_text = render_reflection(feedback.get("reflection"))
    if reflection_text:
        lines.append(reflection_text)
    return "\n".join(lines)


def render_reflection(reflection: Mapping[str, Any] | None) -> str:
    """Render the optional LLM reflection (``agent.reflection``) for the prompt."""
    if not reflection:
        return ""
    titles = {
        "observations": "Observations",
        "hypotheses": "Hypotheses to test next",
        "avoid": "Avoid",
    }
    lines = ["Reviewer notes on the last batch (LLM interpretation of the evidence above):"]
    for key, title in titles.items():
        items = reflection.get(key) or []
        if items:
            lines.append(f"  {title}:")
            lines.extend(f"    - {item}" for item in items)
    return "\n".join(lines) if len(lines) > 1 else ""
