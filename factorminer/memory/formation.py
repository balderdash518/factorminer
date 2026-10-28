"""Memory Formation operator F(M, tau).

Analyzes a mining trajectory tau (batch of evaluated candidates with IC,
correlation, admission results) and extracts new experience:
- Successful patterns from admitted factors
- Forbidden directions from high-correlation rejections
- Strategic insights about what works across the batch
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from factorminer.memory.feedback import (
    FEEDBACK_HISTORY_LIMIT,
    history_row,
    is_implementation_failure,
    is_redundancy_failure,
    summarize_trajectory,
    update_family_ledger,
)
from factorminer.memory.memory_store import (
    ExperienceMemory,
    ForbiddenDirection,
    MiningState,
    StrategicInsight,
    SuccessPattern,
)

# ---------------------------------------------------------------------------
# Operator-pattern detection helpers
# ---------------------------------------------------------------------------

# Named pattern families. Each family lists anchor groups; a formula belongs
# to the family only if it contains at least one token from *every* group.
# Operators match by name or name prefix (``TsLinReg`` -> ``TsLinRegSlope``),
# features match exactly. Names follow the paper's Tables 6/10 so learned
# entries stay readable; formulas outside every family get a structural name.
_PATTERN_SIGNATURES: dict[str, list[list[str]]] = {
    "Higher Moment Regimes": [["Skew", "Kurt"], ["IfElse"]],
    "PV Corr Interaction": [["Corr", "Cov"], ["$volume", "$amt"]],
    "Robust Efficiency": [["Med", "Median"]],
    "Smoothed Efficiency Rank": [["EMA"], ["CsRank"]],
    "Trend Regression Adaptive": [["Rsquare", "Slope", "Resi", "TsLinReg"]],
    "Logical Or Extreme Regimes": [["Or", "And"], ["Greater", "Less"]],
    "Kurtosis Regime": [["Kurt"], ["IfElse"]],
    "Amt Efficiency Rank Interaction": [["$amt"], ["CsRank"]],
}

_FORBIDDEN_SIGNATURES: dict[str, dict[str, Any]] = {
    "Standardized Returns/Amount": {
        "keywords": [["CsZScore"], ["$returns", "$amt"]],
        "typical_corr": 0.6,
        "reason": "Standardized return/amount variants cluster with rho > 0.6",
    },
    "VWAP Deviation variants": {
        "keywords": [["$vwap"], ["Sub", "Delta", "Div"]],
        "typical_corr": 0.5,
        "reason": "VWAP deviation variants produce highly correlated factors (rho > 0.5)",
    },
    "Simple Delta Reversal": {
        "keywords": [["Delta"], ["Neg"]],
        "typical_corr": 0.5,
        "reason": "Simple delta-based reversal factors are redundant (rho > 0.5)",
    },
    "WMA/EMA Smoothed Efficiency": {
        "keywords": [["WMA", "EMA", "SMA"]],
        "typical_corr": 0.9,
        "reason": "WMA/EMA smoothed efficiency variants nearly identical (rho > 0.9)",
    },
}


def _extract_operators(formula: str) -> list[str]:
    """Extract operator names from a DSL formula string."""
    return re.findall(r"([A-Z][a-zA-Z]+)\(", formula)


def _extract_features(formula: str) -> list[str]:
    """Extract feature references from a DSL formula string."""
    return re.findall(r"\$[a-z]+", formula)


def _matches_pattern(formula: str, anchor_groups: list[list[str]]) -> bool:
    """True when the formula contains a token from every anchor group."""
    ops = set(_extract_operators(formula))
    feats = set(_extract_features(formula))

    def present(token: str) -> bool:
        if token.startswith("$"):
            return token in feats
        return any(op == token or op.startswith(token) for op in ops)

    return bool(anchor_groups) and all(
        any(present(token) for token in group) for group in anchor_groups
    )


def _classify_success_pattern(formula: str) -> str | None:
    """Try to classify a formula into a known success pattern category."""
    for pattern_name, keywords in _PATTERN_SIGNATURES.items():
        if _matches_pattern(formula, keywords):
            return pattern_name
    return None


def _classify_forbidden_direction(formula: str) -> str | None:
    """Try to classify a formula into a known forbidden direction."""
    for direction_name, info in _FORBIDDEN_SIGNATURES.items():
        if _matches_pattern(formula, info["keywords"]):
            return direction_name
    return None


# ---------------------------------------------------------------------------
# Trajectory analysis
# ---------------------------------------------------------------------------

def _analyze_admissions(
    trajectory: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Split trajectory into admitted and rejected candidates."""
    admitted = []
    rejected = []
    for candidate in trajectory:
        if candidate.get("admitted", False):
            admitted.append(candidate)
        else:
            rejected.append(candidate)
    return admitted, rejected


def _extract_success_patterns(
    admitted: list[dict],
    existing_patterns: list[SuccessPattern],
) -> list[SuccessPattern]:
    """Extract this batch's success-pattern evidence from admitted factors.

    Only the batch delta is returned (occurrence counts from this batch);
    the evolution operator adds it to ``existing_patterns``. Seeding the map
    with existing counts made every batch double the accumulated totals.
    """
    del existing_patterns  # accumulated state is merged by evolve_memory
    pattern_map: dict[str, SuccessPattern] = {}

    for candidate in admitted:
        formula = candidate.get("formula", "")
        factor_id = candidate.get("factor_id", formula[:60])
        ic = candidate.get("paper_ic", candidate.get("ic", 0.0)) or 0.0
        icir = candidate.get("paper_icir", candidate.get("icir", 0.0)) or 0.0
        max_corr = candidate.get("max_correlation", 0.0) or 0.0

        pattern_name = _classify_success_pattern(formula)
        if pattern_name is None:
            # Novel pattern: create a generic entry based on operators used
            ops = _extract_operators(formula)
            if len(ops) >= 2:
                pattern_name = f"Novel: {'+'.join(ops[:3])}"
            else:
                continue

        if pattern_name in pattern_map:
            pat = pattern_map[pattern_name]
            pat.occurrence_count += 1
            if factor_id not in pat.example_factors:
                pat.example_factors.append(factor_id)
                # Keep example list bounded
                if len(pat.example_factors) > 10:
                    pat.example_factors = pat.example_factors[-10:]
            # Upgrade success rate if consistently passing
            if pat.occurrence_count >= 5 and pat.success_rate == "Medium":
                pat.success_rate = "High"
        else:
            pattern_map[pattern_name] = SuccessPattern(
                name=pattern_name,
                description=(
                    f"Pattern derived from admitted factor with |IC|={ic:.4f}, "
                    f"ICIR={icir:.2f}, max library rho={max_corr:.2f}"
                ),
                template=formula,
                success_rate="Low",
                example_factors=[factor_id],
                occurrence_count=1,
            )

    return list(pattern_map.values())


def _extract_forbidden_directions(
    rejected: list[dict],
    existing_forbidden: list[ForbiddenDirection],
) -> list[ForbiddenDirection]:
    """Extract this batch's redundancy evidence (delta only, see above)."""
    del existing_forbidden  # accumulated state is merged by evolve_memory
    direction_map: dict[str, ForbiddenDirection] = {}

    for candidate in rejected:
        formula = candidate.get("formula", "")
        rejection_reason = candidate.get("rejection_reason", "")
        max_corr = candidate.get("max_correlation", 0.0) or 0.0
        correlated_with = candidate.get("correlated_with", "") or candidate.get(
            "duplicate_of", ""
        )

        # P_fail holds regions rejected for redundancy (paper §3.3); weak or
        # broken formulas are reported through batch feedback instead.
        if not is_redundancy_failure(candidate):
            continue

        direction_name = _classify_forbidden_direction(formula)
        if direction_name is None:
            # Generic high-correlation cluster named by its leading structure
            ops = _extract_operators(formula)
            feats = list(dict.fromkeys(_extract_features(formula)))
            direction_name = f"HighCorr: {'+'.join(ops[:2])}({','.join(feats[:2])})"

        if direction_name in direction_map:
            d = direction_map[direction_name]
            d.occurrence_count += 1
            if correlated_with and correlated_with not in d.correlated_factors:
                d.correlated_factors.append(correlated_with)
                if len(d.correlated_factors) > 10:
                    d.correlated_factors = d.correlated_factors[-10:]
            # Update typical correlation as running average
            if max_corr > 0:
                d.typical_correlation = (
                    d.typical_correlation * (d.occurrence_count - 1) + max_corr
                ) / d.occurrence_count
        else:
            direction_map[direction_name] = ForbiddenDirection(
                name=direction_name,
                description=f"e.g. {formula[:120]} -- rejected: {rejection_reason}",
                correlated_factors=[correlated_with] if correlated_with else [],
                typical_correlation=max_corr,
                reason=rejection_reason or f"High correlation (rho={max_corr:.2f})",
                occurrence_count=1,
            )

    return list(direction_map.values())


_MIN_TRAIT_ATTEMPTS = 5  # valid attempts needed on each side of a comparison
_MIN_RATE_GAP = 0.2  # admission-rate difference worth reporting


def _derive_insights(
    admitted: list[dict],
    rejected: list[dict],
    batch_number: int,
) -> list[StrategicInsight]:
    """Derive higher-level strategic insights from a batch."""
    insights: list[StrategicInsight] = []
    if not admitted and not rejected:
        return insights

    # Formulas that never ran carry no evidence about operators or features.
    broken = [c for c in rejected if is_implementation_failure(c)]
    rejected = [c for c in rejected if not is_implementation_failure(c)]
    total = len(admitted) + len(rejected)
    admission_rate = len(admitted) / total if total > 0 else 0.0

    if len(broken) >= 3:
        broken_ops = Counter(
            op for c in broken for op in set(_extract_operators(c.get("formula", "")))
        )
        if broken_ops:
            op, count = broken_ops.most_common(1)[0]
            insights.append(StrategicInsight(
                insight=f"Formulas using '{op}' often fail to evaluate; check its arity and parameters",
                evidence=f"Batch {batch_number}: {count}/{len(broken)} failed formulas used {op}",
                batch_source=batch_number,
            ))

    # Insight: overall batch admission rate (a fact about this batch only)
    if total >= 5:
        if admission_rate > 0.3:
            insights.append(StrategicInsight(
                insight="Current direction is productive with high admission rate",
                evidence=f"Batch {batch_number}: {len(admitted)}/{total} admitted ({admission_rate:.0%})",
                batch_source=batch_number,
            ))
        elif admission_rate < 0.05:
            insights.append(StrategicInsight(
                insight="Admission rate fell below 5% in this batch",
                evidence=f"Batch {batch_number}: only {len(admitted)}/{total} admitted ({admission_rate:.0%})",
                batch_source=batch_number,
            ))

    # Operator, feature and non-linearity insights compare admission *rates*
    # among all valid attempts with and without the trait. Counting how many
    # admitted factors carry a trait only describes what was generated: if
    # nearly every candidate is non-linear, so is nearly every admission.
    valid = [(c, True) for c in admitted] + [(c, False) for c in rejected]

    def rate_gap(has_trait) -> tuple[int, int, int, int] | None:
        with_trait = [ok for c, ok in valid if has_trait(c)]
        without = [ok for c, ok in valid if not has_trait(c)]
        if len(with_trait) < _MIN_TRAIT_ATTEMPTS or len(without) < _MIN_TRAIT_ATTEMPTS:
            return None
        return sum(with_trait), len(with_trait), sum(without), len(without)

    def rate_insight(label: str, has_trait) -> None:
        gap = rate_gap(has_trait)
        if gap is None:
            return
        hit_w, n_w, hit_wo, n_wo = gap
        rate_w, rate_wo = hit_w / n_w, hit_wo / n_wo
        if rate_w - rate_wo < _MIN_RATE_GAP:
            return
        insights.append(StrategicInsight(
            insight=f"{label} had a higher admission rate than the rest of this batch",
            evidence=(
                f"Batch {batch_number}: {hit_w}/{n_w} ({rate_w:.0%}) with vs "
                f"{hit_wo}/{n_wo} ({rate_wo:.0%}) without; single batch, same panel"
            ),
            batch_source=batch_number,
        ))

    op_sets = {id(c): set(_extract_operators(c.get("formula", ""))) for c, _ in valid}
    feat_sets = {id(c): set(_extract_features(c.get("formula", ""))) for c, _ in valid}
    admitted_ops = Counter(op for c in admitted for op in op_sets[id(c)])
    for op, _ in admitted_ops.most_common(5):
        rate_insight(f"Formulas using '{op}'", lambda c, op=op: op in op_sets[id(c)])
    admitted_feats = Counter(f for c in admitted for f in feat_sets[id(c)])
    for feat, _ in admitted_feats.most_common(3):
        rate_insight(f"Formulas using '{feat}'", lambda c, feat=feat: feat in feat_sets[id(c)])

    nonlinear_ops = {"IfElse", "Skew", "Kurt", "Square", "Pow", "Log", "Or", "And"}
    rate_insight(
        "Formulas with non-linear operators (IfElse/Skew/Kurt/Square/Pow/Log/Or/And)",
        lambda c: bool(op_sets[id(c)] & nonlinear_ops),
    )

    return insights


# ---------------------------------------------------------------------------
# Public API: Memory Formation
# ---------------------------------------------------------------------------

def form_memory(
    memory: ExperienceMemory,
    trajectory: list[dict],
    batch_number: int = 0,
) -> ExperienceMemory:
    """Memory Formation operator F(M, tau).

    Analyzes the mining trajectory tau and forms new experience memories
    to be merged into the existing memory via the evolution operator.

    Parameters
    ----------
    memory : ExperienceMemory
        Current memory state (used for context, not modified in place).
    trajectory : list[dict]
        Batch of evaluated candidates. Each dict should contain:
        - formula: str - the DSL formula
        - factor_id: str - unique identifier
        - ic: float - information coefficient
        - icir: float - IC information ratio
        - max_correlation: float - max correlation with existing factors
        - correlated_with: str - ID of most correlated existing factor
        - admitted: bool - whether the factor was admitted to the library
        - rejection_reason: str - reason for rejection (if rejected)
    batch_number : int
        Current batch/iteration number.

    Returns
    -------
    ExperienceMemory
        A *new* ExperienceMemory containing only the newly formed entries
        (to be merged by the evolution operator).
    """
    admitted, rejected = _analyze_admissions(trajectory)
    feedback = summarize_trajectory(
        trajectory, batch_number, history=memory.state.feedback_history
    )

    # Extract patterns
    new_success = _extract_success_patterns(admitted, memory.success_patterns)
    new_forbidden = _extract_forbidden_directions(rejected, memory.forbidden_directions)
    new_insights = _derive_insights(admitted, rejected, batch_number)

    # Build updated mining state
    new_state = MiningState(
        library_size=memory.state.library_size + len(admitted),
        recent_admissions=[
            {
                "factor_id": c.get("factor_id", ""),
                "formula": c.get("formula", ""),
                "ic": c.get("ic", 0.0),
                "batch": batch_number,
            }
            for c in admitted
        ],
        recent_rejections=[
            {
                "factor_id": c.get("factor_id", ""),
                "formula": c.get("formula", ""),
                "reason": c.get("rejection_reason", ""),
                "max_correlation": c.get("max_correlation", 0.0),
                "batch": batch_number,
            }
            for c in rejected[-20:]  # Keep only last 20 rejections
        ],
        domain_saturation=_compute_domain_saturation(
            memory.state.domain_saturation, admitted, rejected
        ),
        admission_log=memory.state.admission_log + [
            {
                "batch": batch_number,
                "admitted": len(admitted),
                "rejected": len(rejected),
                "admission_rate": len(admitted) / max(len(trajectory), 1),
            }
        ],
        last_batch_feedback=feedback,
        feedback_history=(
            memory.state.feedback_history + [history_row(feedback)]
        )[-FEEDBACK_HISTORY_LIMIT:],
        family_ledger=update_family_ledger(
            memory.state.family_ledger, trajectory, batch_number
        ),
    )

    return ExperienceMemory(
        state=new_state,
        success_patterns=new_success,
        forbidden_directions=new_forbidden,
        insights=new_insights,
        version=memory.version,
    )


def _compute_domain_saturation(
    existing_saturation: dict[str, float],
    admitted: list[dict],
    rejected: list[dict],
) -> dict[str, float]:
    """Compute per-category domain saturation metrics.

    Saturation increases when many candidates in a category are rejected
    due to high correlation (the domain is "full").
    """
    saturation = dict(existing_saturation)

    # Count category attempts and rejections
    category_attempts: dict[str, int] = defaultdict(int)
    category_rejections: dict[str, int] = defaultdict(int)

    for candidate in admitted + rejected:
        if is_implementation_failure(candidate):
            continue
        formula = candidate.get("formula", "")
        category = _classify_success_pattern(formula) or "Other"
        category_attempts[category] += 1

    for candidate in rejected:
        formula = candidate.get("formula", "")
        if is_redundancy_failure(candidate):
            category = _classify_success_pattern(formula) or "Other"
            category_rejections[category] += 1

    # Update saturation with exponential moving average
    alpha = 0.3
    for category, attempts in category_attempts.items():
        if attempts > 0:
            batch_saturation = category_rejections.get(category, 0) / attempts
            old = saturation.get(category, 0.0)
            saturation[category] = (1 - alpha) * old + alpha * batch_saturation

    return saturation
