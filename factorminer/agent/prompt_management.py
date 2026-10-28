"""Versioned chat templates for the traced mining LLM calls.

Publishing is explicit (scripts/sync_langfuse_prompts.py); mining only reads
the production label. Local prompts remain the fallback when offline.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from factorminer.agent.prompt_builder import PromptBuilder
from factorminer.utils.tracing import get_tracer

logger = logging.getLogger(__name__)

REPAIR_USER_TEMPLATE = (
    "The following factor formulas failed to parse. "
    "Fix each one so it uses ONLY valid operators and features "
    "from the library. Return them in the same numbered format:\n"
    "<number>. <name>: <corrected_formula>\n\n"
    "Broken formulas:\n{{broken_formulas}}"
    "\n\nFix all syntax errors, unknown operators, and invalid "
    "feature names. Every formula must be a valid nested function "
    "call using only operators from the library."
)

REFLECTION_USER_TEMPLATE = (
    "{{evidence}}\n\nCurrent library formulas:\n{{library_formulas}}\n\n"
    "Return ONLY a JSON object with these keys:\n"
    '  "observations": up to 4 short statements of what the evidence shows,\n'
    '  "hypotheses": up to 4 testable ideas for the next batch, each naming the '
    "mechanism and why it should be both predictive and uncorrelated with the library,\n"
    '  "avoid": up to 4 structures or neighbourhoods the next batch should not repeat.\n'
    "Each item is one sentence (max 40 words)."
)


def render_template(template: str, variables: dict[str, str]) -> str:
    """Substitute once, leaving braces inside dynamic evidence untouched."""
    return re.sub(r"\{\{(\w+)\}\}", lambda m: variables[m[1]], template)


def prompt_definitions() -> dict[str, dict[str, Any]]:
    """Inventory and publication payloads; no network access or credentials."""
    from factorminer.agent.evidence_reviewer import REVIEW_SYSTEM_PROMPT
    from factorminer.agent.reflection import REFLECTION_SYSTEM_PROMPT

    generation_system = PromptBuilder().system_prompt
    definitions = {}
    for observation, system, user, source in (
        ("propose-factors", generation_system, "{{batch_context}}", "agent/prompt_builder.py"),
        ("repair-formulas", generation_system, REPAIR_USER_TEMPLATE, "agent/factor_generator.py"),
        ("reflect-on-batch", REFLECTION_SYSTEM_PROMPT, REFLECTION_USER_TEMPLATE, "agent/reflection.py"),
        ("review-similar-attempts", REVIEW_SYSTEM_PROMPT, "{{review_context}}", "agent/evidence_reviewer.py"),
    ):
        definitions[observation] = {
            "name": f"factorminer/{observation}",
            "type": "chat",
            "prompt": [
                {"type": "message", "role": "system", "content": system},
                {"type": "message", "role": "user", "content": user},
            ],
            "labels": ["production"],
            "tags": ["factorminer"],
            "config": {
                "source": f"factorminer/{source}",
                "local_system_sha256": hashlib.sha256(system.encode()).hexdigest(),
            },
        }
    return definitions


def resolve_prompt(
    observation: str,
    system: str,
    user: str,
    variables: dict[str, str],
) -> tuple[str, str, Any]:
    """Compile a managed prompt and return the exact version to link to tracing.

    Custom/specialist systems and changed feature registries keep their local
    prompts, rather than silently losing their operator or asset context.
    """
    definition = prompt_definitions().get(observation)
    if definition is None or system != definition["prompt"][0]["content"]:
        return system, user, None
    managed = get_tracer().get_prompt(definition["name"])
    if managed is None:
        return system, user, None
    try:
        if managed.config.get("local_system_sha256") != definition["config"]["local_system_sha256"]:
            raise ValueError("managed prompt does not match the local feature/system contract")
        required = set(re.findall(r"\{\{(\w+)\}\}", str(managed.prompt)))
        if not required <= variables.keys():
            raise ValueError("managed prompt requires unavailable variables")
        messages = managed.compile(**variables)
        if len(messages) != 2 or [m.get("role") for m in messages] != ["system", "user"]:
            raise ValueError("provider expects exactly one system and one user message")
        if not all(isinstance(m.get("content"), str) for m in messages):
            raise ValueError("managed prompt content must be text")
        return messages[0]["content"], messages[1]["content"], managed
    except Exception as exc:
        logger.warning("Using local prompt for %s (%s)", definition["name"], type(exc).__name__)
        return system, user, None
