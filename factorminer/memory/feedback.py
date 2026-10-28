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

import copy
import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

PARSE_ERROR = "parse_error"
COMPUTE_ERROR = "compute_error"
INVALID_SIGNAL = "invalid_signal"
LOW_IC = "low_ic"
LOW_ICIR = "low_icir"
FAILED_CHECKS = "failed_checks"  # evaluation.portfolio_checks (Sharpe, turnover, ...)
REDUNDANT = "redundant"
BATCH_DUPLICATE = "batch_duplicate"
RESEARCH_REJECTED = "research_rejected"
ADMITTED = "admitted"
REPLACED = "replaced"
REJECTED = "rejected"  # legacy entry without enough detail to classify

IMPLEMENTATION_OUTCOMES = frozenset({PARSE_ERROR, COMPUTE_ERROR, INVALID_SIGNAL})
QUALITY_OUTCOMES = frozenset({LOW_IC, LOW_ICIR, FAILED_CHECKS})
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


def _portfolio(entry: Mapping[str, Any]) -> dict[str, float]:
    """Book metrics from evaluation.portfolio_checks, rounded for the prompt."""
    book = entry.get("portfolio") or {}
    keys = ("sharpe", "fitness", "turnover", "returns", "sub_universe_sharpe")
    return {key: round(float(book[key]), 3) for key in keys if book.get(key) is not None}


def _weak_rank(entry: Mapping[str, Any]) -> float:
    """Sharpe when the book was simulated (the binding gate), else |IC|."""
    book = entry.get("portfolio") or {}
    return float(book["sharpe"]) if book.get("sharpe") is not None else _quality(entry)


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
        (e for e in entries if e["_outcome"] in QUALITY_OUTCOMES), key=_weak_rank, reverse=True
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
                **({"portfolio": _portfolio(e)} if _portfolio(e) else {}),
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
    failed_checks = [e for e in entries if e["_outcome"] == FAILED_CHECKS]
    if failed_checks:
        tally = Counter(
            reason.split(" ")[0] if not reason.startswith("Sub-universe") else "Sub-universe"
            for e in failed_checks
            for reason in str(e.get("rejection_reason", "")).split("; ")
            if reason
        )
        sharpes = sorted(_weak_rank(e) for e in failed_checks)
        notes.append(
            f"{len(failed_checks)}/{evaluated} failed the portfolio checks "
            f"(median Sharpe {sharpes[len(sharpes) // 2]:.2f}); failing checks: "
            + ", ".join(f"{name} {count}" for name, count in tally.most_common())
            + ". Sharpe ~ IC x sqrt(breadth): a steadier IC sign matters more than a larger mean."
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
        ADMITTED, REPLACED, REDUNDANT, BATCH_DUPLICATE, LOW_IC, LOW_ICIR, FAILED_CHECKS,
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
        gate = "portfolio checks" if any(i.get("portfolio") for i in feedback["weak"]) else "IC gate"
        lines.append(f"Closest to the {gate} but rejected:")
        for item in feedback["weak"]:
            book = item.get("portfolio") or {}
            book_text = (
                f"; Sharpe {book['sharpe']:.2f}, Fitness {book.get('fitness', 0.0):.2f}, "
                f"Turnover {book.get('turnover', 0.0):.0%}"
                if "sharpe" in book else ""
            )
            lines.append(
                f"  - {item['formula']}  (|IC| {item['ic']:.4f}, ICIR {item['icir']:.2f}{book_text})"
            )
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


# ---------------------------------------------------------------------------
# Cumulative per-family ledger
# ---------------------------------------------------------------------------

_LEDGER_EXAMPLES = 2


def _formula_key(formula: str) -> str:
    return hashlib.sha1("".join(str(formula).split()).encode("utf-8")).hexdigest()[:12]


def _empty_ledger_row() -> dict[str, Any]:
    return {
        "tested": 0, "admitted": 0, "redundant": 0, "weak": 0, "broken": 0,
        "evaluated": 0, "ic_sum": 0.0, "repeats": 0, "last_iteration": 0,
        "best": [], "seen": [],
    }


def update_family_ledger(
    ledger: Mapping[str, Mapping[str, Any]] | None,
    trajectory: Iterable[Mapping[str, Any]],
    iteration: int,
) -> dict[str, dict[str, Any]]:
    """Fold one batch into the cumulative per-family record (returns a new dict).

    Families come from :func:`factorminer.architecture.families.infer_family`.
    Each row keeps outcome counts, the |IC| sum over formulas that ran, the
    best formulas so far and short hashes of every formula, so an exact
    resubmission of an earlier formula is counted as a repeat.
    """
    from factorminer.architecture.families import infer_family

    updated = {family: copy.deepcopy(dict(row)) for family, row in (ledger or {}).items()}
    for entry in trajectory:
        formula = str(entry.get("formula", "") or "")
        if not formula:
            continue
        outcome = candidate_outcome(entry)
        row = updated.setdefault(infer_family(formula), _empty_ledger_row())
        key = _formula_key(formula)
        if key in row["seen"]:
            row["repeats"] += 1
        else:
            row["seen"].append(key)
        row["tested"] += 1
        row["last_iteration"] = int(iteration)
        if outcome in SUCCESS_OUTCOMES:
            row["admitted"] += 1
        elif outcome in REDUNDANCY_OUTCOMES:
            row["redundant"] += 1
        elif outcome in IMPLEMENTATION_OUTCOMES:
            row["broken"] += 1
            continue
        else:
            row["weak"] += 1
        ic = _quality(entry)
        row["evaluated"] += 1
        row["ic_sum"] = round(row["ic_sum"] + ic, 6)
        if all(b["formula"] != _short(formula, 120) for b in row["best"]):
            row["best"].append({
                "formula": _short(formula, 120),
                "ic": round(ic, 4),
                "iteration": int(iteration),
                "outcome": outcome,
            })
            row["best"] = sorted(row["best"], key=lambda b: b["ic"], reverse=True)[:_LEDGER_EXAMPLES]
    return updated


def render_family_ledger(
    ledger: Mapping[str, Mapping[str, Any]] | None,
    available_features: Iterable[str] | None = None,
) -> str:
    """Render the cumulative per-family record as prompt text.

    ``available_features`` hides untried families the dataset cannot express
    (e.g. VWAP without ``$vwap``); ``None`` assumes every leaf exists.
    """
    if not ledger:
        return ""
    from factorminer.architecture.families import FAMILY_REQUIRED_FEATURE, FINE_FAMILIES

    rows = sorted(ledger.items(), key=lambda item: (-int(item[1].get("tested", 0)), item[0]))
    total = sum(int(row.get("tested", 0)) for _, row in rows)
    lines = [f"=== EXPLORATION HISTORY BY FAMILY (all batches, {total} candidates) ==="]
    for family, row in rows:
        parts = [f"{row['tested']} tested", f"{row['admitted']} admitted"]
        parts += [f"{row[key]} {key}" for key in ("redundant", "weak", "broken") if row.get(key)]
        line = f"- {family}: {', '.join(parts)}"
        if row.get("evaluated"):
            mean = float(row["ic_sum"]) / int(row["evaluated"])
            line += f"; |IC| best {row['best'][0]['ic']:.4f}, mean {mean:.4f}"
        lines.append(f"{line}; last it{row.get('last_iteration', '?')}")
        for best in row.get("best", []):
            lines.append(f"    {best['formula']}  (|IC| {best['ic']:.4f}, it{best['iteration']}, {best['outcome']})")
    repeats = sum(int(row.get("repeats", 0)) for _, row in rows)
    if repeats:
        lines.append(
            f"{repeats} candidate(s) exactly repeated an earlier formula; "
            "re-testing a formula returns the same numbers."
        )
    available = set(available_features) if available_features is not None else None
    untried = [
        family for family in FINE_FAMILIES
        if family not in ledger
        and (available is None or FAMILY_REQUIRED_FEATURE.get(family, "") in available | {""})
    ]
    if untried:
        lines.append(f"No candidates yet in: {', '.join(untried)}")
    return "\n".join(lines)
