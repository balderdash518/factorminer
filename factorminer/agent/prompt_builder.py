"""Build prompts for LLM-driven factor generation using memory priors.

The system prompt encodes the full operator library, syntax rules, feature
list, and task description.  The user prompt injects per-iteration context:
memory signals, library state, and output format instructions.
"""

from __future__ import annotations

import re
from typing import Any

from factorminer.core.types import (
    OPERATOR_REGISTRY,
    OperatorSpec,
    get_feature_descriptions,
    get_features,
)


def _format_operator_table() -> str:
    """Build a human-readable operator reference table grouped by category."""
    grouped: dict[str, list[OperatorSpec]] = {}
    for spec in OPERATOR_REGISTRY.values():
        cat = spec.category.name
        grouped.setdefault(cat, []).append(spec)

    lines: list[str] = []
    for cat_name in [
        "ARITHMETIC",
        "STATISTICAL",
        "TIMESERIES",
        "SMOOTHING",
        "CROSS_SECTIONAL",
        "REGRESSION",
        "LOGICAL",
        "AUTO_INVENTED",
    ]:
        specs = grouped.get(cat_name, [])
        if not specs:
            continue
        lines.append(f"\n### {cat_name} operators")
        for spec in sorted(specs, key=lambda s: s.name):
            params_str = ""
            if spec.param_names:
                parts = []
                for pname in spec.param_names:
                    default = spec.param_defaults.get(pname, "")
                    lo, hi = spec.param_ranges.get(pname, (None, None))
                    range_str = f"[{lo}-{hi}]" if lo is not None else ""
                    parts.append(f"{pname}={default}{range_str}")
                params_str = f"  params: {', '.join(parts)}"
            arity_args = ", ".join([f"expr{i+1}" for i in range(spec.arity)])
            if spec.param_names:
                arity_args += ", " + ", ".join(spec.param_names)
            lines.append(f"- {spec.name}({arity_args}): {spec.description}{params_str}")
    return "\n".join(lines)


def _format_feature_list(features: list[str] | None = None) -> str:
    """Build a description of available raw features.

    Uses the live feature registry so extra leaves (fundamentals, futures)
    appear in generation prompts once registered. Descriptions come from
    :data:`factorminer.core.types.FEATURE_DESCRIPTIONS`.
    """
    names = list(features) if features is not None else get_features()
    descriptions = get_feature_descriptions(names)
    lines = []
    for feat in names:
        desc = descriptions.get(feat, "")
        if desc:
            lines.append(f"  {feat}: {desc}")
        else:
            lines.append(f"  {feat}")
    return "\n".join(lines)


def _asset_class_framing(asset_class: str | None = None) -> str:
    """Return the opening sentence for the system prompt by asset class."""
    kind = (asset_class or "equity").strip().lower()
    if kind in {"futures", "future", "continuous_futures"}:
        return (
            "You are a quantitative researcher mining formulaic alpha factors "
            "for cross-sectional futures / continuous-contract selection."
        )
    if kind in {"crypto", "cryptocurrency", "digital_asset"}:
        return (
            "You are a quantitative researcher mining formulaic alpha factors "
            "for cryptocurrency cross-sectional selection."
        )
    if kind in {"multi_asset", "multi-asset", "mixed"}:
        return (
            "You are a quantitative researcher mining formulaic alpha factors "
            "across a multi-asset panel."
        )
    return (
        "You are a quantitative researcher mining formulaic alpha factors "
        "for stock selection."
    )


# (formula, explanation) pairs shown to the model; an example is dropped when
# the active feature set lacks one of its leaves.
_EXAMPLE_FACTORS: tuple[tuple[str, str], ...] = (
    ("Neg(CsRank(Delta($close, 5)))",
     "Short-term reversal: rank of 5-day price change, negated."),
    ("CsZScore(Div(Sub($volume, Mean($volume, 20)), Std($volume, 20)))",
     "Volume surprise: standardized deviation from 20-day mean volume."),
    ("CsRank(Div(Sub($close, $vwap), $vwap))",
     "Intraday deviation from VWAP, cross-sectionally ranked."),
    ("Neg(Corr($volume, $close, 10))",
     "Negative price-volume correlation over 10 days."),
    ("CsRank(TsLinRegSlope($volume, 20))",
     "Trend in trading volume over 20 days, ranked."),
    ("IfElse(Greater($returns, 0), Std($returns, 10), Neg(Std($returns, 10)))",
     "Conditional volatility: positive for up-moves, negative for down-moves."),
    ("CsRank(Div(Sub($close, TsMin($low, 20)), Sub(TsMax($high, 20), TsMin($low, 20))))",
     "Position within 20-day price range, ranked."),
)
_FEATURE_IN_TEXT = re.compile(r"\$[a-zA-Z_]+")


def _format_examples(names: list[str]) -> str:
    available = set(names)
    return "\n".join(
        f"- {formula}\n  {explanation}"
        for formula, explanation in _EXAMPLE_FACTORS
        if set(_FEATURE_IN_TEXT.findall(formula)) <= available
    )


def _underused_features_hint(names: list[str]) -> str:
    underused = [f"{name}, " for name in ("$amt", "$vwap") if name in names]
    return f"({''.join(underused)}and any registered alt-data leaves are underused)"


def build_system_prompt(
    *,
    features: list[str] | None = None,
    asset_class: str | None = None,
) -> str:
    """Build the generation system prompt for the active feature set.

    Parameters
    ----------
    features :
        Optional explicit DSL leaf list. Defaults to the live registry
        (``get_features()``), so callers that ``register_features(...)``
        before building prompts automatically surface new leaves.
    asset_class :
        Optional framing hint (``equity`` / ``futures`` / ``crypto``).
        Defaults to equity/stock-selection language for backward compatibility.
    """
    names = list(features) if features is not None else get_features()
    feature_block = _format_feature_list(names)
    framing = _asset_class_framing(asset_class)
    return f"""{framing}

Your goal is to generate novel, predictive factor expressions using a tree-structured domain-specific language (DSL). Each factor is a composition of operators applied to raw market features.

## RAW FEATURES (leaf nodes)
{feature_block}

## OPERATOR LIBRARY
{_format_operator_table()}

## EXPRESSION SYNTAX RULES
1. Every expression is a nested function call: Operator(args...)
2. Leaf nodes are raw features ($close, $volume, etc.) or numeric constants.
3. Operators are called by name with expression arguments first, then numeric parameters:
   - Mean($close, 20) = rolling mean of $close over the last 20 bars
   - Corr($close, $volume, 10) = rolling correlation of close and volume over 10 bars
   - IfElse(Greater($returns, 0), $volume, Neg($volume)) = conditional
4. No infix operators; use Add(x,y) instead of x+y, Sub(x,y) instead of x-y, etc.
5. Parameters like window sizes are trailing numeric arguments after expression children.
6. Valid window sizes are integers counted in bars of the data frequency; check each operator's parameter ranges above.
7. Cross-sectional operators (CsRank, CsZScore, CsDemean, CsScale, CsNeutralize) operate across all assets at each time step -- they are crucial for making factors comparable.

## EXAMPLES OF WELL-FORMED FACTORS
{_format_examples(names)}

## KEY PRINCIPLES FOR HIGH-QUALITY FACTORS
- Evaluation uses cross-sectional rank IC, so an outer CsRank/CsZScore is optional; use one when it changes the meaning (e.g. before combining terms of different scale).
- Combine DIFFERENT operator types for novelty (e.g., time-series + cross-sectional + arithmetic).
- Use diverse window sizes; avoid always defaulting to 10.
- Explore uncommon feature combinations {_underused_features_hint(names)}.
- Factors with depth 3-7 tend to be best: deep enough to capture non-trivial patterns but not so deep they overfit.
- Prefer economically meaningful combinations over random nesting.
"""


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

# Default system prompt — rebuilt from the live feature registry at import
# time. Callers that register extra leaves should prefer
# :func:`build_system_prompt` (or pass a custom string into PromptBuilder).
SYSTEM_PROMPT = build_system_prompt()


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------

def normalize_factor_references(entries: list[Any] | None) -> list[str]:
    """Convert mixed factor metadata into prompt-safe string references."""
    if not entries:
        return []

    normalized: list[str] = []
    seen: set[str] = set()

    for entry in entries:
        text = ""
        if isinstance(entry, str):
            text = entry.strip()
        elif isinstance(entry, dict):
            formula = str(entry.get("formula", "")).strip()
            name = str(entry.get("name", "")).strip()
            category = str(entry.get("category", "")).strip()
            if formula and name:
                text = f"{name}: {formula}"
            elif formula:
                text = formula
            elif name and category:
                text = f"{name} [{category}]"
            elif name:
                text = name
        elif entry is not None:
            text = str(entry).strip()

        if text and text not in seen:
            normalized.append(text)
            seen.add(text)

    return normalized


_LIBRARY_MEMBER_LIMIT = 40


def _library_sections(library_state: dict[str, Any]) -> list[str]:
    """Render library size, composition and members for the user prompt."""
    lib_size = library_state.get("library_size", library_state.get("size", 0))
    target = library_state.get("target_size", 110)
    sections = [
        f"\n## CURRENT LIBRARY STATUS\n"
        f"Library size: {lib_size} / {target} factors."
    ]
    threshold = library_state.get("correlation_threshold")
    if threshold is not None:
        sections.append(
            f"A candidate is admitted only if |IC| >= {library_state.get('ic_threshold', '?')} "
            f"and its correlation with every library factor is below {threshold}."
        )
    if library_state.get("portfolio_checks"):
        sections.append(str(library_state["portfolio_checks"]))

    categories = library_state.get("categories") or {}
    if categories:
        ordered = sorted(categories.items(), key=lambda item: (-item[1], item[0]))
        sections.append(
            "Composition by category: "
            + ", ".join(f"{name}={count}" for name, count in ordered)
        )

    members = library_state.get("members") or []
    if members:
        shown = members[-_LIBRARY_MEMBER_LIMIT:]
        header = "Library factors (new candidates must be uncorrelated with these)"
        if len(members) > len(shown):
            header += f", most recent {len(shown)} of {len(members)}"
        lines = []
        for member in shown:
            ic = member.get("ic_paper_mean")
            ic_text = f"  |IC| {float(ic):.4f}" if ic is not None else ""
            lines.append(f"  #{member.get('id', '?')} {member.get('formula', '')}{ic_text}")
        sections.append(header + ":\n" + "\n".join(lines))
    else:
        recent = normalize_factor_references(library_state.get("recent_admissions", []))
        if recent:
            sections.append(
                "Recently admitted factors:\n"
                + "\n".join(f"  - {f}" for f in recent[-10:])
            )

    saturation = library_state.get("domain_saturation", {})
    if saturation:
        sat_lines = [f"  {domain}: {pct:.0%} saturated" for domain, pct in saturation.items()]
        sections.append("Domain saturation:\n" + "\n".join(sat_lines))
    return sections


def _describe_memory_entry(entry: Any) -> str:
    """Render one memory entry (string or pattern dict) as a prompt line."""
    if not isinstance(entry, dict):
        return str(entry)
    name = str(entry.get("name") or entry.get("insight") or "").strip()
    details = []
    for key in ("description", "reason", "evidence"):
        value = str(entry.get(key, "") or "").strip()
        if value and value != name:
            details.append(value)
    template = str(entry.get("template", "") or "").strip()
    if template:
        details.append(f"template: {template}")
    examples = entry.get("example_factors") or entry.get("correlated_factors") or []
    if examples:
        details.append("e.g. " + ", ".join(str(e) for e in examples[:3]))
    return name + (f" -- {'; '.join(details)}" if details else "")


def _structured_memory_sections(memory_signal: dict[str, Any]) -> list[str]:
    """Fallback rendering when the memory signal carries no ``prompt_text``."""
    sections: list[str] = []
    rec_dirs = memory_signal.get("recommended_directions", [])
    if rec_dirs:
        sections.append(
            "\n## RECOMMENDED DIRECTIONS (focus on these successful patterns)\n"
            + "\n".join(f"  * {_describe_memory_entry(d)}" for d in rec_dirs)
        )
    forbidden = memory_signal.get("forbidden_directions", [])
    if forbidden:
        sections.append(
            "\n## FORBIDDEN DIRECTIONS (AVOID these -- they produce correlated/weak factors)\n"
            + "\n".join(f"  X {_describe_memory_entry(d)}" for d in forbidden)
        )
    insights = memory_signal.get("strategic_insights") or memory_signal.get("insights") or []
    if insights:
        sections.append(
            "\n## STRATEGIC INSIGHTS\n"
            + "\n".join(f"  Note: {_describe_memory_entry(ins)}" for ins in insights)
        )
    rejections = memory_signal.get("recent_rejections", [])
    if rejections:
        rej_lines = []
        for rej in rejections[-10:]:
            name = rej.get("name") or rej.get("formula") or "unknown"
            reason = rej.get("reason", "unknown")
            rej_lines.append(f"  - {name}: rejected because {reason}")
        sections.append(
            "\n## RECENT REJECTIONS (learn from these failures)\n"
            + "\n".join(rej_lines)
        )
    return sections


class PromptBuilder:
    """Constructs system and user prompts for factor generation.

    The system prompt is static (operator library + rules).
    The user prompt varies each iteration based on memory signals.
    """

    def __init__(
        self,
        system_prompt: str | None = None,
        *,
        features: list[str] | None = None,
        asset_class: str | None = None,
    ) -> None:
        if system_prompt is not None:
            self._system_prompt = system_prompt
        else:
            self._system_prompt = build_system_prompt(
                features=features,
                asset_class=asset_class,
            )
        self._features = list(features) if features is not None else None
        self._asset_class = asset_class

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def features(self) -> list[str] | None:
        """Leaves the prompt offers; ``None`` means the live registry."""
        return list(self._features) if self._features is not None else None

    def build_user_prompt(
        self,
        memory_signal: dict[str, Any],
        library_state: dict[str, Any],
        batch_size: int = 40,
    ) -> str:
        """Build the per-iteration user prompt injecting memory priors.

        Parameters
        ----------
        memory_signal : dict
            Keys:
            - ``"prompt_text"`` : str -- rendered memory (P_succ, P_fail, I and
              last-batch feedback); preferred when present
            - ``"recommended_directions"`` / ``"forbidden_directions"`` :
              list[str | dict] -- used only without ``prompt_text``
            - ``"insights"`` (or ``"strategic_insights"``) : list[str | dict]
            - ``"recent_rejections"`` : list[dict] -- recent rejection reasons
        library_state : dict
            Keys (``FactorLibrary.get_state_summary`` plus ``target_size``):
            - ``"library_size"`` (or legacy ``"size"``) : int
            - ``"target_size"`` : int -- target library size
            - ``"categories"`` : dict[str, int] -- factors per category
            - ``"members"`` : list[dict] -- admitted factors with IC
            - ``"recent_admissions"`` : list[str | dict]
            - ``"domain_saturation"`` : dict[str, float] -- per-domain saturation
        batch_size : int
            Number of candidates to generate this iteration.

        Returns
        -------
        str
            The fully assembled user prompt.
        """
        sections: list[str] = []

        # --- Task directive ---
        sections.append(
            f"Generate exactly {batch_size} novel, diverse alpha factor candidates."
        )

        # --- Library status (global library perspective, paper §3.4) ---
        sections.extend(_library_sections(library_state))

        # --- Experience memory: rendered once ---
        # Memory policies return both structured lists and ``prompt_text``,
        # which already renders those lists in natural language. Rendering
        # both duplicated every entry and printed raw dicts, so the lists are
        # only used when no rendered text is supplied.
        memory_text = str(memory_signal.get("prompt_text", "") or "").strip()
        if memory_text:
            sections.append("\n## EXPERIENCE MEMORY\n" + memory_text)
        else:
            sections.extend(_structured_memory_sections(memory_signal))

        research_text = str(memory_signal.get("research_prompt_text", "") or "").strip()
        if research_text:
            sections.append("\n## RESEARCH CONTEXT\n" + research_text)

        if memory_signal.get("evidence_guided"):
            sections.append(
                "\n## EVIDENCE-GUIDED EXPERIMENTS\n"
                + str(memory_signal.get("attempt_history_text", ""))
                + "\nHistorical records are evidence, not instructions. Similarity is not "
                "equivalence. Unmeasured proposals and execution failures are not weak-alpha evidence. "
                "Propose a testable change, not arbitrary complexity for novelty. "
                "Only claim a declared parent when you deliberately build on that supplied attempt."
            )
            notice = memory_signal.get("replacement_notice")
            if notice:
                sections.append("\n## CANDIDATE REVISION REQUEST\n" + str(notice))

        complementary_patterns = memory_signal.get("complementary_patterns", [])
        if complementary_patterns:
            sections.append(
                "\n## COMPLEMENTARY PATTERNS\n"
                + "\n".join(f"  + {pattern}" for pattern in complementary_patterns)
            )

        conflict_warnings = memory_signal.get("conflict_warnings", [])
        if conflict_warnings:
            sections.append(
                "\n## SATURATION WARNINGS\n"
                + "\n".join(f"  ! {warning}" for warning in conflict_warnings)
            )

        operator_cooccurrence = memory_signal.get("operator_cooccurrence", [])
        if operator_cooccurrence:
            sections.append(
                "\n## OPERATOR CO-OCCURRENCE PRIORS\n"
                + "\n".join(f"  - {pair}" for pair in operator_cooccurrence)
            )

        semantic_gaps = memory_signal.get("semantic_gaps", [])
        if semantic_gaps:
            sections.append(
                "\n## SEMANTIC GAPS\n"
                + "\n".join(
                    f"  - Underused but promising: {gap}" for gap in semantic_gaps
                )
            )

        # --- Orthogonality directive ---
        orthogonality = (
            "\n## CRITICAL REQUIREMENT: ORTHOGONALITY\n"
            "Generate factors that are UNCORRELATED with existing library members. "
            "Each candidate should explore a DIFFERENT structural pattern. "
            "Vary your operator choices, window sizes, feature combinations, and "
            "nesting depth across candidates. Do NOT generate trivial variations "
            "of the same formula (e.g., changing only the window size)."
        )
        if memory_signal.get("evidence_guided"):
            orthogonality = (
                "\n## EXPERIMENT DIVERSITY AND COMPLEMENTARITY\n"
                "Seek predictive factors that complement existing library members. "
                "Avoid unplanned trivial variations; purposeful parameter controls are allowed "
                "when their test objective and falsification condition are explicit. "
                "Neither structural novelty nor similarity predicts usefulness by itself."
            )
        sections.append(orthogonality)

        # --- Output format ---
        if memory_signal.get("evidence_guided"):
            sections.append(
                f"\n## OUTPUT FORMAT\nReturn a JSON array of exactly {batch_size} experiments. "
                "Each object has name, formula, hypothesis, reference_ids (supplied attempt IDs), "
                "declared_parent_id (optional), change_description, expected_improvement, "
                "and falsification_condition. Do not invent evidence IDs or statistics. "
                "For a reviewer-only similarity warning you may retain a purposeful controlled "
                "experiment by supplying retention_reason and a concrete falsifiable test. "
                "This never overrides a deterministic duplicate/invalid-formula rejection. "
                "Use only the available DSL features and operators. No markdown or extra text."
            )
            return "\n".join(sections)
        sections.append(
            f"\n## OUTPUT FORMAT\n"
            f"Output exactly {batch_size} factors, one per line.\n"
            f"Format each line as: <number>. <factor_name>: <formula>\n"
            f"Example:\n"
            f"1. momentum_reversal: Neg(CsRank(Delta($close, 5)))\n"
            f"2. volume_surprise: CsZScore(Div(Sub($volume, Mean($volume, 20)), Std($volume, 20)))\n"
            f"\nRules:\n"
            f"- factor_name: lowercase_with_underscores, descriptive, unique\n"
            f"- formula: valid DSL expression using ONLY operators and features listed above\n"
            f"- No markdown, no explanations, no extra text -- just the numbered list\n"
            f"- Every formula must parse correctly with the operator library"
        )

        return "\n".join(sections)


# ---------------------------------------------------------------------------
# New specialist/critic/debate prompt builder functions
# ---------------------------------------------------------------------------

def build_specialist_prompt(
    specialist_name: str,
    specialist_domain: str,
    specialist_hypothesis: str,
    preferred_operators: list[str],
    preferred_features: list[str],
    example_factors: list[str],
    avoid_patterns: list[str],
    memory_signal: dict[str, Any] | None = None,
    library_diagnostics: dict[str, Any] | None = None,
    regime_context: str = "",
    n_proposals: int = 15,
    success_rate: float | None = None,
) -> str:
    """Build a rich context-aware user prompt for a specialist agent.

    Parameters
    ----------
    specialist_name : str
        Human-readable name of the specialist (e.g. ``"MomentumMiner"``).
    specialist_domain : str
        Short domain description for the specialist.
    specialist_hypothesis : str
        Core economic hypothesis guiding this specialist.
    preferred_operators : list[str]
        Operator names this specialist should lean on.
    preferred_features : list[str]
        Feature names this specialist prefers.
    example_factors : list[str]
        Reference formula examples for this specialist.
    avoid_patterns : list[str]
        Structural patterns to explicitly avoid.
    memory_signal : dict or None
        Experience memory context (recommended directions, etc.).
    library_diagnostics : dict or None
        Library state (size, saturation, recent admissions).
    regime_context : str
        Current market regime description.
    n_proposals : int
        Number of proposals to request.
    success_rate : float or None
        Historical success rate for this specialist (for context).

    Returns
    -------
    str
        Fully assembled specialist user prompt.
    """
    memory_signal = memory_signal or {}
    library_diagnostics = library_diagnostics or {}
    sections: list[str] = []

    # Header
    sections.append(
        f"## SPECIALIST TASK: {specialist_name}\n"
        f"Domain: {specialist_domain}\n"
        f"Hypothesis: {specialist_hypothesis}"
    )

    if success_rate is not None:
        sections.append(
            f"Your historical admission rate: {success_rate:.1%}  "
            f"(aim to exceed this by proposing higher-quality factors)"
        )

    # Regime context
    if regime_context:
        sections.append(
            f"\n## CURRENT MARKET REGIME\n{regime_context}"
        )

    # Library state
    lib_size = library_diagnostics.get("size", 0)
    target = library_diagnostics.get("target_size", 110)
    sections.append(
        f"\n## LIBRARY STATUS\nCurrent: {lib_size}/{target} factors."
    )

    recent = normalize_factor_references(
        library_diagnostics.get("recent_admissions", [])
    )
    if recent:
        sections.append(
            "Recently admitted (avoid similar patterns):\n"
            + "\n".join(f"  - {f}" for f in recent[-8:])
        )

    saturation = library_diagnostics.get("domain_saturation", {})
    if saturation:
        sat_lines = [
            f"  {d}: {p:.0%} saturated" for d, p in saturation.items()
        ]
        sections.append("Domain saturation:\n" + "\n".join(sat_lines))

    # Memory signal injections
    rec_dirs = memory_signal.get("recommended_directions", [])
    if rec_dirs:
        sections.append(
            "\n## RECOMMENDED DIRECTIONS\n"
            + "\n".join(f"  * {d}" for d in rec_dirs)
        )

    forbidden = memory_signal.get("forbidden_directions", [])
    if forbidden:
        sections.append(
            "\n## FORBIDDEN DIRECTIONS\n"
            + "\n".join(f"  X {d}" for d in forbidden)
        )

    insights = memory_signal.get("strategic_insights", [])
    if insights:
        sections.append(
            "\n## STRATEGIC INSIGHTS\n"
            + "\n".join(f"  - {ins}" for ins in insights)
        )

    helix_text = memory_signal.get("prompt_text", "").strip()
    if helix_text:
        sections.append(f"\n## HELIX CONTEXT\n{helix_text}")
    if memory_signal.get("evidence_guided"):
        sections.append("\n## HISTORICAL EXPERIMENT EVIDENCE\n" + str(memory_signal.get("attempt_history_text", "")))

    comp_patterns = memory_signal.get("complementary_patterns", [])
    if comp_patterns:
        sections.append(
            "\n## COMPLEMENTARY PATTERNS (explore these)\n"
            + "\n".join(f"  + {p}" for p in comp_patterns)
        )

    warn = memory_signal.get("conflict_warnings", [])
    if warn:
        sections.append(
            "\n## SATURATION WARNINGS\n"
            + "\n".join(f"  ! {w}" for w in warn)
        )

    gaps = memory_signal.get("semantic_gaps", [])
    if gaps:
        sections.append(
            "\n## SEMANTIC GAPS (underused areas to explore)\n"
            + "\n".join(f"  ~ {g}" for g in gaps)
        )

    # Specialist focus directive
    ops_str = ", ".join(preferred_operators)
    feats_str = ", ".join(preferred_features)
    sections.append(
        f"\n## YOUR SPECIALIST FOCUS\n"
        f"Preferred operators: {{{ops_str}}}\n"
        f"Preferred features: {{{feats_str}}}\n"
        f"Focus ~60% of proposals on these.  The remaining ~40% should "
        f"explore creative cross-domain combinations."
    )

    # Domain examples
    if example_factors:
        sections.append(
            "\n## DOMAIN REFERENCE EXAMPLES (structural templates, do NOT copy exactly)\n"
            + "\n".join(f"  - {ex}" for ex in example_factors)
        )

    # Avoid patterns
    if avoid_patterns:
        sections.append(
            "\n## PATTERNS TO AVOID\n"
            + "\n".join(f"  X {av}" for av in avoid_patterns)
        )

    # Few-shot patterns from memory
    mem_success_patterns = memory_signal.get("_few_shot_examples", [])
    if mem_success_patterns:
        sections.append(
            "\n## FEW-SHOT SUCCESS PATTERNS FROM MEMORY\n"
            "(These formulas were previously admitted -- use as structural inspiration)\n"
            + "\n".join(f"  [+] {ex}" for ex in mem_success_patterns[:5])
        )

    # Output format
    sections.append(
        f"\n## OUTPUT FORMAT\n"
        f"Generate exactly {n_proposals} novel factor candidates.\n"
        f"Format: <number>. <factor_name>: <formula>\n"
        f"Example: 1. momentum_reversal: Neg(CsRank(Delta($close, 5)))\n"
        f"Rules:\n"
        f"- factor_name: lowercase_with_underscores, unique, descriptive\n"
        f"- formula: valid DSL expression only\n"
        f"- No markdown, no explanations -- just the numbered list\n"
        f"- Every formula must use only registered operators and features"
    )

    return "\n".join(sections)


def build_critic_scoring_prompt(
    candidates: list[dict[str, str]],
    existing_factors: list[str] | None = None,
    memory_signal: str | None = None,
    regime_context: str = "",
) -> str:
    """Build a structured JSON-output scoring prompt for the critic agent.

    Parameters
    ----------
    candidates : list[dict]
        List of dicts with keys ``"name"``, ``"formula"``, ``"specialist"``.
    existing_factors : list[str] or None
        Formula strings already in the library.
    memory_signal : str or None
        Free-text memory context (success patterns, etc.).
    regime_context : str
        Current market regime description.

    Returns
    -------
    str
        Fully assembled critic scoring prompt.
    """
    existing_factors = existing_factors or []
    sections: list[str] = []

    sections.append(
        "## CRITIC SCORING TASK\n"
        "Evaluate the following candidate factors for economic intuition.\n"
        "Score each on how well it captures a plausible, economically "
        "meaningful cross-sectional return predictor."
    )

    if regime_context:
        sections.append(f"\n## CURRENT REGIME\n{regime_context}")

    if existing_factors:
        sections.append(
            "\n## LIBRARY SAMPLE (existing factors to avoid duplicating)\n"
            + "\n".join(f"  - {f}" for f in existing_factors[-12:])
        )

    if memory_signal:
        sections.append(f"\n## MEMORY CONTEXT (success patterns)\n{memory_signal[:600]}")

    sections.append("\n## CANDIDATES")
    for c in candidates:
        name = c.get("name", "unknown")
        formula = c.get("formula", "")
        specialist = c.get("specialist", "unknown")
        sections.append(
            f"  [{specialist}] {name}: {formula}"
        )

    sections.append(
        "\n## SCORING CRITERIA\n"
        "economic_intuition [0.0-1.0]:\n"
        "  1.0 = strong economic story, appropriate complexity, novel signal\n"
        "  0.5 = plausible but generic or overly simple\n"
        "  0.0 = no coherent economic story, trivial, or clearly wrong\n"
        "\nConsider:\n"
        "  - Is there a coherent alpha story (momentum, reversal, vol, liquidity)?\n"
        "  - Is complexity appropriate (depth 3-7, 3-5 unique operators)?\n"
        "  - Does it use features in a semantically meaningful way?\n"
        "  - Is it structurally distinct from existing library members?\n"
        "  - Would a quant researcher find this plausible?"
    )

    sections.append(
        "\n## OUTPUT FORMAT\n"
        "One JSON object per line for each candidate:\n"
        '{"name": "<factor_name>", "economic_intuition": <0.0-1.0>, '
        '"rationale": "<one concise sentence>"}\n'
        "Output ONLY the JSON lines. No markdown, no extra text."
    )

    return "\n".join(sections)


def build_debate_synthesis_prompt(
    all_proposals: list[dict[str, Any]],
    critic_scores: list[dict[str, Any]],
    top_k: int = 10,
) -> str:
    """Build a consensus synthesis prompt for the debate orchestrator.

    Used when a final synthesis step is desired to resolve conflicts
    between specialist proposals and produce a consensus ranking.

    Parameters
    ----------
    all_proposals : list[dict]
        All proposals with ``"name"``, ``"formula"``, ``"specialist"`` keys.
    critic_scores : list[dict]
        Critic scores with ``"name"`` and ``"composite_score"`` keys.
    top_k : int
        Number of top factors to synthesize consensus for.

    Returns
    -------
    str
        Debate synthesis prompt.
    """
    # Sort by composite score
    score_map = {s["name"]: s.get("composite_score", 0.5) for s in critic_scores}
    sorted_proposals = sorted(
        all_proposals,
        key=lambda p: score_map.get(p.get("name", ""), 0.0),
        reverse=True,
    )[:top_k * 2]  # take 2x top_k for synthesis

    sections: list[str] = []
    sections.append(
        f"## DEBATE SYNTHESIS TASK\n"
        f"Multiple specialist agents proposed the following factors.\n"
        f"The critic has pre-scored them.  Your task is to identify the "
        f"top {top_k} most complementary factors for a diverse library.\n"
        f"Prioritize NOVELTY and ORTHOGONALITY over pure individual quality."
    )

    sections.append("\n## SCORED PROPOSALS (sorted by critic score)")
    for p in sorted_proposals:
        name = p.get("name", "?")
        formula = p.get("formula", "?")
        specialist = p.get("specialist", "?")
        score = score_map.get(name, 0.5)
        sections.append(
            f"  [{specialist}, score={score:.2f}] {name}: {formula}"
        )

    sections.append(
        f"\n## SELECTION CRITERIA\n"
        f"Select the top {top_k} factors that are:\n"
        f"  1. Diverse in operator structure (avoid near-duplicates)\n"
        f"  2. Balanced across specialist domains where possible\n"
        f"  3. High composite critic score\n"
        f"  4. Economically interpretable\n"
        f"\nOutput a ranked list: <rank>. <factor_name>\n"
        f"No other text."
    )

    return "\n".join(sections)
