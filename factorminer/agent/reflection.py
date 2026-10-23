"""Optional LLM reflection over a batch's deterministic feedback.

RD-Agent's research loop ends each round with an LLM summary (observations,
hypothesis evaluation, next hypothesis). FactorMiner's memory is rule-based,
so the generator receives evidence but no interpretation of it. This module
adds that interpretation as a separate, optional step:

* input: the batch feedback rendered by :mod:`factorminer.memory.feedback`
  (all numbers computed by the evaluator) plus the current library formulas;
* output: short observations, testable hypotheses for the next batch and
  directions to avoid, each of which must cite the evidence it rests on.

The reflection never changes admission or memory statistics; it is stored
next to the feedback record and rendered into the next prompt. Unparseable
responses are dropped.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from factorminer.agent.llm_interface import provider_call_info
from factorminer.agent.prompt_management import (
    REFLECTION_USER_TEMPLATE,
    render_template,
    resolve_prompt,
)
from factorminer.utils.tracing import get_tracer

logger = logging.getLogger(__name__)

REFLECTION_SYSTEM_PROMPT = (
    "You are the research reviewer in a formulaic alpha mining loop. You read the "
    "evaluator's results for one batch of candidate factors and decide what the next "
    "batch should test. Every claim must rest on the numbers you are given; do not "
    "invent statistics. Separate formulas that failed to run from ideas that ran and "
    "were judged weak or redundant."
)

_MAX_ITEMS = 4


def build_reflection_prompt(evidence: str, library_formulas: list[str] | None = None) -> str:
    library_block = "\n".join(f"  - {f}" for f in (library_formulas or [])[-30:]) or "  (empty)"
    return render_template(
        REFLECTION_USER_TEMPLATE, {"evidence": evidence, "library_formulas": library_block},
    )


def parse_reflection(text: str) -> dict[str, list[str]] | None:
    """Extract the reflection JSON object; ``None`` when absent or malformed."""
    match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    if match is None:
        return None
    try:
        payload = json.loads(match.group(0))
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    result: dict[str, list[str]] = {}
    for key in ("observations", "hypotheses", "avoid"):
        values = payload.get(key, [])
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            continue
        cleaned = [str(v).strip()[:300] for v in values if str(v).strip()]
        if cleaned:
            result[key] = cleaned[:_MAX_ITEMS]
    return result or None


def reflect_on_batch(
    provider: Any,
    evidence: str,
    *,
    library_formulas: list[str] | None = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> dict[str, list[str]] | None:
    """Ask the LLM to interpret one batch; returns ``None`` on any failure."""
    if not evidence.strip():
        return None
    prompt = build_reflection_prompt(evidence, library_formulas)
    system, prompt, managed = resolve_prompt(
        "reflect-on-batch", REFLECTION_SYSTEM_PROMPT, prompt,
        {"evidence": evidence, "library_formulas": "\n".join(
            f"  - {f}" for f in (library_formulas or [])[-30:]
        ) or "  (empty)"},
    )
    with get_tracer().generation(
        "reflect-on-batch",
        model=getattr(provider, "model", None),
        system_prompt=system,
        user_prompt=prompt,
        prompt=managed,
        model_parameters={"temperature": temperature, "max_tokens": max_tokens},
    ) as generation:
        try:
            raw = provider.generate(
                system_prompt=system,
                user_prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # reflection is advisory; never break mining
            logger.warning("Batch reflection failed: %s", exc)
            generation.update(level="WARNING", status_message=f"reflection failed: {exc}")
            return None
        call = provider_call_info(provider)
        reflection = parse_reflection(raw)
        generation.update(
            output=raw,
            model=call.get("model"),
            usage_details=call.get("usage") or None,
            metadata={"parsed": reflection is not None},
        )
    if reflection is None:
        logger.debug("Batch reflection returned no parseable JSON")
    return reflection
