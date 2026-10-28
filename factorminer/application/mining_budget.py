"""Mining budget and candidate-evaluation contracts."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Budget Tracker
# ---------------------------------------------------------------------------


@dataclass
class BudgetTracker:
    """Tracks resource consumption across the mining session.

    Retains the legacy limit fields for compatibility with existing callers.
    Mining uses these counters for observation and ignores exhaustion flags.
    """

    max_llm_calls: int = 0  # 0 = unlimited
    max_wall_seconds: float = 0  # 0 = unlimited

    # Running totals
    llm_calls: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    compute_seconds: float = 0.0
    wall_start: float = field(default_factory=time.time)

    def record_llm_call(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        self.llm_calls += 1
        self.llm_prompt_tokens += prompt_tokens
        self.llm_completion_tokens += completion_tokens

    def record_compute(self, seconds: float) -> None:
        self.compute_seconds += seconds

    @property
    def wall_elapsed(self) -> float:
        return time.time() - self.wall_start

    @property
    def total_tokens(self) -> int:
        return self.llm_prompt_tokens + self.llm_completion_tokens

    def is_exhausted(self) -> bool:
        """Legacy diagnostic only; mining does not stop on this value."""
        if self.max_llm_calls > 0 and self.llm_calls >= self.max_llm_calls:
            return True
        if self.max_wall_seconds > 0 and self.wall_elapsed >= self.max_wall_seconds:
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "llm_calls": self.llm_calls,
            "llm_prompt_tokens": self.llm_prompt_tokens,
            "llm_completion_tokens": self.llm_completion_tokens,
            "total_tokens": self.total_tokens,
            "compute_seconds": round(self.compute_seconds, 2),
            "wall_elapsed_seconds": round(self.wall_elapsed, 2),
        }


# ---------------------------------------------------------------------------
# Candidate evaluation result
# ---------------------------------------------------------------------------

# Candidate outcomes. Implementation failures (parse/compute/invalid) say the
# formula did not run as written; quality and redundancy failures say it ran
# and was judged. Feedback and memory keep the two apart.
OUTCOME_PARSE_ERROR = "parse_error"
OUTCOME_COMPUTE_ERROR = "compute_error"
OUTCOME_INVALID_SIGNAL = "invalid_signal"
OUTCOME_LOW_IC = "low_ic"
OUTCOME_LOW_ICIR = "low_icir"
OUTCOME_FAILED_CHECKS = "failed_checks"  # evaluation.portfolio_checks
OUTCOME_REDUNDANT = "redundant"
OUTCOME_BATCH_DUPLICATE = "batch_duplicate"
OUTCOME_RESEARCH_REJECTED = "research_rejected"
OUTCOME_ADMITTED = "admitted"
OUTCOME_REPLACED = "replaced"

IMPLEMENTATION_FAILURES = frozenset(
    {OUTCOME_PARSE_ERROR, OUTCOME_COMPUTE_ERROR, OUTCOME_INVALID_SIGNAL}
)
REDUNDANCY_FAILURES = frozenset({OUTCOME_REDUNDANT, OUTCOME_BATCH_DUPLICATE})
OUTCOMES = (
    OUTCOME_PARSE_ERROR,
    OUTCOME_COMPUTE_ERROR,
    OUTCOME_INVALID_SIGNAL,
    OUTCOME_LOW_IC,
    OUTCOME_LOW_ICIR,
    OUTCOME_FAILED_CHECKS,
    OUTCOME_REDUNDANT,
    OUTCOME_BATCH_DUPLICATE,
    OUTCOME_RESEARCH_REJECTED,
    OUTCOME_ADMITTED,
    OUTCOME_REPLACED,
)


@dataclass
class EvaluationResult:
    """Result of evaluating a single candidate factor."""

    factor_name: str
    formula: str
    parse_ok: bool = False
    ic_mean: float = 0.0
    ic_paper_mean: float = 0.0
    ic_abs_mean: float = 0.0
    icir: float = 0.0
    ic_paper_icir: float = 0.0
    ic_win_rate: float = 0.0
    max_correlation: float = 0.0
    correlated_with: str = ""  # nearest library factor (by dependence)
    conflict_count: int = 0  # library factors at or above the correlation threshold
    admitted: bool = False
    replaced: int | None = None  # ID of replaced factor, if any
    rejection_reason: str = ""
    stage_passed: int = 0  # 0=parse/IC fail, 1=IC pass, 2=corr pass, 3=dedup pass, 4=admitted
    # Structured outcome (one of OUTCOMES) so feedback never has to parse reason text.
    outcome: str = ""
    replacement_note: str = ""  # why Stage 2.5 did or did not replace
    duplicate_of: str = ""  # batch member kept instead of this candidate (Stage 3)
    signals: np.ndarray | None = None
    target_stats: dict[str, dict] = field(default_factory=dict)
    research_score: float = 0.0
    research_lcb: float = 0.0
    residual_ic: float = 0.0
    projection_loss: float = 0.0
    effective_rank_gain: float = 0.0
    log_det_gain: float = 0.0
    score_vector: dict[str, Any] | None = None
    portfolio: dict[str, float] | None = None  # evaluation.portfolio_checks metrics
    # Parent-formula lineage for EditAwareMemoryPolicy + MRM developmental history.
    parent_formula: str = ""
    parent_ic_paper_mean: float | None = None
    edit_type: str = ""
    edit_motif: str = ""
    secondary_parent_formula: str = ""
    # Deliberate experiment ancestry is separate from inferred similarity.
    attempt_id: str = ""
    declared_parent_id: str = ""
    nearest_prior_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
