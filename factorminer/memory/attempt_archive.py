"""Run-scoped proposal/evidence archive and explainable historical retrieval.

This is deliberately separate from the market-contact trial ledger: a proposal,
rule rejection or LLM opinion is not a numerical observation. Exact identities
preserve executor AST semantics; neither SymPy nor embeddings authorize vetoes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

from factorminer.core.expression_tree import ConstantNode, LeafNode, OperatorNode
from factorminer.core.parser import try_parse
from factorminer.memory.retrieval import (
    bm25_scores,
    dense_cosine_scores,
    reciprocal_rank_fusion,
    scores_to_ranks,
)

ARCHIVE_FILENAME = "attempt_archive.jsonl"
IDENTITY_VERSION = "executor-ordered-ast-v1"
_SUCCESS = {"admitted", "replaced"}
_BROKEN = {"parse_error", "compute_error", "invalid_signal"}
_LIBRARY_DEPENDENT = {"admitted", "replaced", "redundant", "batch_duplicate", "research_rejected"}
_META_TEXT = (
    "hypothesis", "declared_parent_id", "change_description", "expected_improvement",
    "falsification_condition", "retention_reason",
)


def json_safe(value: Any) -> Any:
    """Keep append-only records strict JSON, including non-finite diagnostics."""
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def clean_proposal_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    metadata = metadata or {}
    result = {k: str(metadata[k])[:1000] for k in _META_TEXT if metadata.get(k)}
    refs = metadata.get("reference_ids", [])
    if isinstance(refs, list):
        result["reference_ids"] = [str(v)[:100] for v in refs[:10]]
    return result


@lru_cache(maxsize=8192)
def formula_descriptor(formula: str) -> dict[str, Any] | None:
    """Ordered, full-precision AST identity plus path-aware retrieval features.

    Do not sort operands, strip Neg/ranks, or use to_string() for numeric identity:
    its display formatting can round constants. Similarity is never equivalence.
    """
    tree = try_parse(formula)
    if tree is None:
        return None
    paths: dict[str, Any] = {}
    tokens: list[str] = []
    subtrees: set[Any] = set()

    def walk(node: Any, path: str) -> tuple[Any, Any]:
        if isinstance(node, LeafNode):
            paths[path] = node.feature_name
            tokens.append(f"{path}:{node.feature_name}")
            return ("feature", node.feature_name), ("feature", node.feature_name)
        if isinstance(node, ConstantNode):
            value = float(node.value).hex()
            paths[path] = node.value
            tokens.append(f"{path}:constant")
            return ("constant", value), ("constant",)
        if not isinstance(node, OperatorNode):
            raise TypeError(f"unsupported executor node: {type(node).__name__}")
        name = node.operator.name
        paths[path] = name
        tokens.append(f"{path}:{name}")
        params = tuple((k, float(v).hex()) for k, v in sorted(node.params.items()))
        for key, value in params:
            paths[f"{path}/param:{key}"] = float.fromhex(value)
        children = [walk(child, f"{path}/{i}") for i, child in enumerate(node.children)]
        skeleton = (name, tuple(c[1] for c in children))
        subtrees.add(skeleton)
        return (
            (name, tuple(c[0] for c in children), params),
            skeleton,
        )

    identity, skeleton = walk(tree.root, "root")
    blob = json.dumps([IDENTITY_VERSION, identity], separators=(",", ":"))
    return {
        "key": hashlib.sha256(blob.encode()).hexdigest(),
        "skeleton": skeleton,
        "tokens": tokens,
        "paths": paths,
        "features": tree.leaf_features(),
        "subtrees": subtrees,
        "size": tree.root.size(),
    }


def library_identity(library_state: Mapping[str, Any]) -> str:
    """Track changing library membership/quality separately from fixed run scope."""
    members = sorted(
        (str(m.get("id", "")), str(m.get("formula", "")), m.get("ic_paper_mean"))
        for m in library_state.get("members", [])
    )
    blob = json.dumps(json_safe(members), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _tokens(text: str) -> list[str]:
    return re.findall(r"\$?\w+", text.lower())


def _role(record: Mapping[str, Any]) -> str:
    if not record.get("measured"):
        return "implementation_issue" if record.get("outcome") in _BROKEN else "untested"
    if record.get("outcome") in _SUCCESS:
        return "success"
    if record.get("outcome") in {"redundant", "batch_duplicate"}:
        return "redundancy"
    return "counterexample"


class AttemptArchive:
    """Append-only events, materialized attempts, and hybrid evidence retrieval."""

    def __init__(self, output_dir: str | Path, scope_id: str, *, embedder: Any = None, load_existing: bool = True):
        self.path = Path(output_dir) / ARCHIVE_FILENAME
        self.scope_id = scope_id
        self.embedder = embedder
        self.attempts: dict[str, dict[str, Any]] = {}
        self._event_ids: set[str] = set()
        self.last_generation_feedback: dict[str, Any] = {}
        self.events: list[dict[str, Any]] = []
        if load_existing and self.path.exists():
            for line_no, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
                try:
                    row = json.loads(line)
                    if row.get("schema_version") != 1:
                        raise ValueError("unsupported attempt archive schema")
                    if row.get("scope_id") == scope_id:
                        self._apply(row)
                except (ValueError, TypeError, KeyError, AttributeError) as exc:
                    raise ValueError(f"Invalid attempt archive at line {line_no}") from exc

    def _apply(self, row: dict[str, Any]) -> None:
        if row["event_id"] in self._event_ids:
            return
        self._event_ids.add(row["event_id"])
        self.events.append(row)
        if row["event"] == "generation_feedback":
            self.last_generation_feedback = dict(row["details"]["feedback"])
        attempt_id = row.get("attempt_id")
        if not attempt_id:
            return
        if row["event"] == "proposed":
            self.attempts[attempt_id] = {
                "attempt_id": attempt_id, "iteration": row["iteration"],
                "measured": False, **row["details"],
            }
        elif attempt_id in self.attempts:
            self.attempts[attempt_id].update(row["details"])
        if attempt_id in self.attempts:
            self.attempts[attempt_id]["status"] = row["event"]

    def event(self, event: str, iteration: int, *, attempt_id: str = "", **details: Any) -> None:
        row = json_safe({
            "schema_version": 1, "scope_id": self.scope_id, "event": event,
            "event_id": uuid4().hex,
            "attempt_id": attempt_id, "iteration": iteration, "details": details,
            "recorded_at": datetime.now(UTC).isoformat(),
        })
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        self._apply(row)

    def import_events(self, source: Path) -> None:
        """Resume a portable checkpoint without overwriting any local history."""
        if not source.exists() or source.resolve() == self.path.resolve():
            return
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
        persisted = {
            json.loads(line)["event_id"]
            for line in self.path.read_text(encoding="utf-8").splitlines()
        } if self.path.exists() else set()
        for row in rows:
            if row.get("schema_version") != 1:
                raise ValueError("Unsupported attempt archive checkpoint schema")
            if row.get("scope_id") != self.scope_id or row["event_id"] in self._event_ids:
                continue
            if row["event_id"] not in persisted:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            self._apply(row)

    def save_snapshot(self, destination: Path) -> None:
        """Only the active history, not future events beyond a resumed checkpoint."""
        if destination.resolve() == self.path.resolve():
            raise ValueError("Checkpoint snapshot must not overwrite the append-only archive")
        with destination.open("w", encoding="utf-8") as handle:
            for row in self.events:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    def propose(self, name: str, formula: str, iteration: int, **details: Any) -> str:
        attempt_id = uuid4().hex
        descriptor = formula_descriptor(formula)
        self.event(
            "proposed", iteration, attempt_id=attempt_id, name=name, formula=formula,
            formula_key=descriptor["key"] if descriptor else "", **details,
        )
        return attempt_id

    def exact_observation(self, key: str, snapshot: str) -> dict[str, Any] | None:
        for record in reversed(list(self.attempts.values())):
            if record.get("formula_key") != key or not record.get("measured"):
                continue
            if record.get("outcome") in _LIBRARY_DEPENDENT and record.get("library_snapshot") != snapshot:
                continue
            return record
        return None

    def retrieve(
        self, query: str, *, formula: str = "", limit: int = 6, exclude_id: str = "",
        diverse: bool = False,
    ) -> list[dict[str, Any]]:
        # One representative per exact formula prevents repeated siblings from
        # consuming the entire evidence budget. Prefer a measured observation.
        representatives: dict[str, dict[str, Any]] = {}
        for record in self.attempts.values():
            key = record.get("formula_key")
            if not key or record["attempt_id"] == exclude_id:
                continue
            old = representatives.get(key)
            if old is None or record.get("measured") or not old.get("measured"):
                representatives[key] = record
        records = list(representatives.values())
        if not records or limit <= 0:
            return []
        descriptor = formula_descriptor(formula) if formula else None
        documents = [f"{r['formula']} {r.get('metadata', {}).get('hypothesis', '')} "
                     f"{r.get('rejection_reason', '')}" for r in records]
        lexical = bm25_scores(_tokens(query), [_tokens(d) for d in documents])
        dense = dense_cosine_scores(query, documents, self.embedder)
        structure = []
        for record in records:
            prior = formula_descriptor(record["formula"])
            if descriptor is None or prior is None:
                structure.append(0.0)
                continue
            left, right = Counter(descriptor["tokens"]), Counter(prior["tokens"])
            score = 2 * sum((left & right).values()) / max(sum(left.values()) + sum(right.values()), 1)
            if descriptor["skeleton"] in prior["subtrees"] or prior["skeleton"] in descriptor["subtrees"]:
                # A wrapper insertion/removal shifts every absolute path, but
                # remains an important near edit. This is still a soft score.
                score = max(score, min(descriptor["size"], prior["size"]) / max(descriptor["size"], prior["size"]))
            structure.append(1.0 if descriptor["skeleton"] == prior["skeleton"] else score)
        channels = [scores_to_ranks(values) for values in (lexical, dense, structure) if any(v > 0 for v in values)]
        fused = reciprocal_rank_fusion(channels) if channels else [0.0] * len(records)
        order = sorted(range(len(records)), key=lambda i: (-fused[i], -records[i]["iteration"]))
        if diverse:
            # Relevant successes + failures + untested/implementation evidence;
            # no invented example when a category has no historical support.
            chosen = []
            for roles in ({"success"}, {"redundancy", "counterexample"}, {"untested", "implementation_issue"}):
                match = next((i for i in order if _role(records[i]) in roles), None)
                if match is not None and len(chosen) < limit:
                    chosen.append(match)
            order = chosen + [i for i in order if i not in chosen]
        hits = []
        for i in order[:limit]:
            record = records[i]
            prior = formula_descriptor(record["formula"])
            differences = []
            if descriptor and prior:
                for path in sorted(descriptor["paths"].keys() | prior["paths"].keys()):
                    old, new = prior["paths"].get(path), descriptor["paths"].get(path)
                    if old != new:
                        differences.append({"path": path, "prior": old, "candidate": new})
            hits.append({
                "attempt_id": record["attempt_id"], "formula": record["formula"],
                "iteration": record["iteration"], "role": _role(record),
                "measured": bool(record.get("measured")), "outcome": record.get("outcome"),
                "hypothesis": record.get("metadata", {}).get("hypothesis", ""),
                "declared_parent_id": record.get("declared_parent_id", ""),
                "change_description": record.get("metadata", {}).get("change_description", ""),
                # Opinions remain explicitly separate from measured results.
                "advisory_review": {
                    key: value[:300] if isinstance(value, str) else value
                    for key, value in record.get("review", {}).items()
                    if key in {"action", "reason", "suggested_test", "status", "evidence_ids"}
                },
                "metrics": record.get("metrics", {}), "reason": record.get("rejection_reason", ""),
                "library_snapshot": record.get("library_snapshot", ""),
                "similarity": {"structural": round(structure[i], 6),
                               "lexical": round(lexical[i], 6), "dense": round(dense[i], 6),
                               "rank_fusion": fused[i], "method": "ordered_ast+bm25+optional_dense"},
                "differences": differences[:10],
            })
        return hits

    def summary(self) -> dict[str, Any]:
        records = list(self.attempts.values())
        return {
            "proposals": len(records),
            "measured": sum(bool(r.get("measured")) for r in records),
            "statuses": dict(Counter(r.get("status", "unknown") for r in records)),
            "observed_outcomes": dict(Counter(r.get("outcome", "unknown") for r in records if r.get("measured"))),
        }
