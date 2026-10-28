# Langfuse v4 migration

Reviewed on 2026-10-24 using the installed Langfuse skill's
`references/v4-project-migration.md` and the [canonical workflow](https://github.com/langfuse/skills/blob/main/skills/langfuse/references/v4-project-migration.md).
Repository changes are complete; project cutover remains blocked on the checks below.
No CLI was installed or run, credentials requested, prompts published, evaluator rules
changed, exports changed, or remote verification traces emitted.

## Scope and compatibility

- Python runtime: repository requires 3.12+, verified locally on 3.12.13.
- `pyproject.toml`: observability extra now declares `langfuse>=5.0.4,<6`.
- `uv.lock`: resolves `langfuse==5.0.4`; PyPI still reports 5.0.4 as the latest stable
  release. The lockfile was regenerated with `uv lock --upgrade-package langfuse`.
- Platform v4 and Python SDK v5 have different version numbers. SDK v5 supports
  Cloud or self-hosted server 4.59.0+. A self-hosted v3/older-v4 deployment must
  upgrade before using this dependency; downgrading the SDK is not this migration.
- The optional SDK remains optional. Runs without it or without credentials retain
  the local JSONL mirror and local prompt fallback.

Sources: [compatibility](https://langfuse.com/docs/compatibility),
[Python SDK v4 to v5](https://langfuse.com/docs/observability/sdk/upgrade-path/python-v4-to-v5),
[project upgrade guide](https://langfuse.com/faq/all/upgrade-to-langfuse-v4).

## Repository inventory and changes

| Surface | Result |
| --- | --- |
| SDK, initialization, lockfile | `pyproject.toml`, `uv.lock`, `MiningTracer._connect` in `factorminer/utils/tracing.py`. Confirmed the current stable SDK; raised the minimum to the verified release. Conflicting host aliases now disable remote tracing instead of silently selecting a different host. |
| Observation hierarchy and correlation | `MiningTracer.iteration` establishes propagation before the root. Root input/output already use observation fields; session, tags, metadata and trace name already propagate. Added explicit run environment propagation so an enclosing propagation context cannot override it. Release remains the SDK client setting. No application user ID or component version is configured; managed prompt name/version are linked to generations. No public-trace setter is used. |
| Attribute producers | `RalphLoop._build_tracer` supplies campaign, dataset, thresholds, batch size and run tags; `tracer_from_config` supplies environment/release/sampling; `LoopServices.execute_iteration` supplies root input/output and numeric scores. Generation call sites attach model, messages, output, usage and managed prompt references. Long metadata now respects SDK v5's JSON-encoded 200-character limit, including escaped characters. |
| Delivery and errors | SDK OTLP batching remains in place. Observation entry/exit failures are contained; exporter teardown cannot suppress or replace a mining exception. `activate` continues to flush on success or failure without shutting down a shared SDK client. The SDK handles retries; this wrapper neither acknowledges delivery nor adds durable replay. SDK shutdown/atexit handles process teardown; abrupt process termination can still lose buffered data. |
| Prompt management and script | `factorminer/agent/prompt_management.py` uses supported cached `get_prompt` and compile APIs. `scripts/sync_langfuse_prompts.py` now references `@langfuse/cli@1.2.4`, checks process exit codes as well as the response envelope, and rejects conflicting hosts. Existing explicit publishing behavior and preview files remain. CLI version/runtime verified from npm (Node.js >=20); subprocess behavior tested without invoking/installing the CLI. |
| Direct APIs and custom ingestion | No raw legacy ingestion, deprecated read endpoints, generated legacy API methods, custom OTLP exporter, or dataset-run API call found in application code, scripts, CI, or notebooks. `create_score` is the supported SDK helper; v5 batches scores through the current scores endpoint. Prompt APIs remain supported. No pagination consumer in repository code needs replacement. |
| Evaluators and exports in code | No Langfuse evaluator-rule, dataset experiment, Blob Storage, Mixpanel or PostHog consumer configuration found. Local factor evaluation/backtesting is independent of Langfuse managed evaluators. Remote state still needs the checks below. |

Sources: [instrumentation](https://langfuse.com/docs/observability/sdk/instrumentation),
[CLI](https://langfuse.com/docs/api-and-data-platform/features/cli),
[evaluator migration](https://langfuse.com/faq/all/llm-as-a-judge-migration).

## Project evidence and limits

Existing `.env` credentials were used in memory through the already installed SDK's
generated API client. No credentials are recorded here. `projects.get` confirmed
project **FactorMiner**, ID `cmvknrfie0001ad0ebga0voea`, on
`https://jp.cloud.langfuse.com`; the project API is accessible.

- Cursor-aware Evaluation Rules and Evaluators API reads returned zero rows and no
  next cursor. No active or inactive rules or referenced definitions were exposed.
  This does **not** establish that the Evaluators UI has no Legacy rows.
- The project Evaluators URL redirected to sign-in in the available browser. The
  required UI inspection could not be completed without a browser session.
- A bounded seven-day Observations v2 read found existing `mining-iteration` roots
  with input keys `iteration`, `library_size`, `batch_size` and output keys
  `admitted`, `outcomes`, `yield_rate`, `library_size`, `research_action`.
  One existing trace contained eight observations, including `propose-factors` and
  `reflect-on-batch` generations; every observation carried the root's session ID.
  These are historical baseline observations, not validation of this change.
- Blob Storage integrations API read returned HTTP 403. Mixpanel, PostHog and other
  integrations could not be inspected through the authenticated UI.
- No separate non-production Langfuse project was confirmed. No live migrated
  traces were emitted and no session-cost aggregation was certified.

There are **no project-verified evaluator migration contracts**. Candidate targets
from code inspection only: the logical root `mining-iteration` (`AGENT`,
`isRootObservation=true`) for iteration summaries, or a single named generation for
prompt/response quality. A root summary does not contain every child's prompt and
output. Inspect actual rule variables, filters, mappings, sampling and representative
payloads before choosing a target; consolidate all required values on that one
observation. Observation evaluators cannot read siblings or children. No dataset
experiment target can be inferred from this repository.

## Verification and rollback

`factorminer/tests/test_langfuse_v4.py` uses the real installed SDK with an in-memory
span exporter and dummy credentials. It verifies root input/output, hierarchy, new
trace IDs per iteration, session on cost-bearing generations, tags, escaped metadata,
environment, release, default private behavior and flushing. Additional tests cover
exporter teardown and flush failures preserving application errors/local logs, host
conflicts, and prompt CLI success/404/error handling. Existing tracing, prompt,
environment, packaging and mining-loop tests exercise surrounding behavior. These
local tests do not verify backend ingestion, project behavior, or delivery retries.

Validation results:

- Focused migration, tracing, prompt management, environment and packaging tests:
  42 passed.
- Additional loop services, Ralph/Helix and external-failure tests: 91 passed,
  1 failed. The unchanged `test_worker_returns_rankings_and_contains_failures`
  expects an `os.abort()` subprocess to return a negative POSIX signal exit code;
  Windows returns a positive exit status and the unchanged worker classifies it
  as `error`. The failure also reproduces in isolation.
- Ruff passes for all changed Python files. Repository-wide Ruff reports the
  pre-existing untracked `scripts/export_qlib_bin.py:96` (`UP017`); that file was
  left untouched.
- `uv lock --check`, `scripts/check_architecture.py`, and `scripts/check.py`
  pass (43 manifest checks, no issues).

Before cutover:

1. Inspect **Legacy** rows in the confirmed project's Evaluators UI, including inactive
   rules without reactivating them. For any active legacy rule, record filters,
   sampling, mappings, payloads, one successor and its cutover test.
2. Create successors disabled; validate newly ingested observations in a confirmed
   non-production project. Obtain approval before enabling. Compare scores/logs and
   cardinality, then disable (preserve) legacy rules for rollback. Dataset targets
   need experiment successors with expected output/metadata mapped to item context.
3. Inspect Project Settings > Integrations, follow each source's documented dual
   transition, and validate schema/path/join/dashboard/history consumers. Preserve
   credentials, schedules, prefixes, formats and field groups; obtain downstream-owner
   confirmation before enriched-only output stops the legacy shape.
4. Run representative migrated iterations at sample rate 1, then inspect Observations
   v2, root I/O, all session IDs, generation usage/cost and total session cost. Re-read
   rules/integrations after approved writes. Review the Cloud migration status page.

Code rollback: revert only the migration diff, retaining unrelated local work, or
set `observability.langfuse: "off"` while preserving local JSONL. The prior SDK was
already 5.0.4; rolling back these changes does not reintroduce legacy SDK ingestion.
No remote state was changed, so there is no project configuration to undo yet.

## Readiness report

| Area | Status | Evidence, blocker and next action |
| --- | --- | --- |
| project access | ready | API access confirmed for FactorMiner on Japan Cloud. Browser access remains limited as noted in the dependent rows. |
| SDK/instrumentation | changed | SDK requirement/lockfile, propagation, metadata and failure isolation updated and locally tested. Blocker: no confirmed non-production backend test; next: ingest representative traces and inspect [observations](https://jp.cloud.langfuse.com/project/cmvknrfie0001ad0ebga0voea/observations). |
| trace evaluators | blocked | Rules/definitions APIs returned zero rows, but UI Legacy inspection requires sign-in. No project-verified contracts; code-only candidate: root or one generation. Next: inspect [Evaluators](https://jp.cloud.langfuse.com/project/cmvknrfie0001ad0ebga0voea/evals) and map/validate any active legacy rules. |
| dataset evaluators | blocked | No API rules or repository dataset evaluator integration found; UI Legacy inspection remains inaccessible. No verified contract or concrete experiment target. Next: inspect dataset legacy rows in [Evaluators](https://jp.cloud.langfuse.com/project/cmvknrfie0001ad0ebga0voea/evals) and prepare disabled experiment successors if needed. |
| direct APIs | ready | Repository scan found no deprecated calls; prompt/score helpers use supported APIs. Existing generated client read confirmed Observations v2 access. |
| exports | blocked | Blob Storage API returned 403; integrations UI requires sign-in. Downstream state is unknown. Next: inspect [Integrations](https://jp.cloud.langfuse.com/project/cmvknrfie0001ad0ebga0voea/settings/integrations), validate consumers and confirm enriched-only cutover with owners. |
| verification/rollback | blocked | Local tests cover the repository changes; historical session baseline inspected. Blocker: no migrated non-production ingestion, session-cost check, UI Legacy audit or export verification. Next: execute the cutover checks above and review [migration status](https://jp.cloud.langfuse.com/v4-migration). Local disable/revert path is documented; remote state is unchanged. |
