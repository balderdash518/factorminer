"""Publish local mining chat templates to Langfuse, without duplicate versions.

Preview: uv run --no-sync python scripts/sync_langfuse_prompts.py
Publish: uv run --no-sync python scripts/sync_langfuse_prompts.py --apply
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from factorminer.agent.prompt_management import prompt_definitions
from factorminer.utils.env_file import load_env_file

LANGFUSE_CLI_PACKAGE = "@langfuse/cli@1.2.4"


def same_template(existing: dict[str, Any], definition: dict[str, Any]) -> bool:
    """The API omits message type; the SDK adds type=message when reading."""
    def messages(prompt):
        return [{k: v for k, v in m.items() if k != "type" or v != "message"} for m in prompt]

    return (
        existing.get("type") == definition["type"]
        and existing.get("config") == definition["config"]
        and messages(existing.get("prompt", [])) == messages(definition["prompt"])
    )


def request_cli(args: list[str], env: dict[str, str]) -> tuple[int, dict[str, Any]]:
    executable = shutil.which("npx.cmd" if os.name == "nt" else "npx")
    if executable is None:
        raise RuntimeError("npx is required to publish Langfuse prompts")
    result = subprocess.run(
        [executable, "--yes", LANGFUSE_CLI_PACKAGE, "api", "prompts", *args, "--json"],
        env=env, capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    try:
        envelope = json.loads(result.stdout)
        status = int(envelope["status"])
        body = envelope.get("body") or {}
        if isinstance(body, str):
            body = json.loads(body)
    except (ValueError, KeyError, TypeError) as exc:
        # Do not echo CLI diagnostics that might include credential material.
        raise RuntimeError("Langfuse CLI did not return a JSON response") from exc
    if status not in (200, 201, 404):
        raise RuntimeError(f"Langfuse prompt request failed (HTTP {status})")
    if result.returncode != 0 and not (result.returncode == 5 and status == 404):
        raise RuntimeError("Langfuse CLI failed to complete the prompt request")
    return status, body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Publish changed templates")
    parser.add_argument("--output-dir", type=Path, default=Path("output/langfuse-prompts"))
    args = parser.parse_args()
    load_env_file()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if (args.apply and env.get("LANGFUSE_BASE_URL") and env.get("LANGFUSE_HOST")
            and env["LANGFUSE_BASE_URL"].rstrip("/") != env["LANGFUSE_HOST"].rstrip("/")):
        parser.error("LANGFUSE_BASE_URL and LANGFUSE_HOST must identify the same host")
    if env.get("LANGFUSE_BASE_URL"):
        env["LANGFUSE_HOST"] = env["LANGFUSE_BASE_URL"]
    if args.apply and not all(env.get(k) for k in (
        "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST",
    )):
        parser.error("Set Langfuse credentials and base URL in .env before publishing")
    for observation, definition in prompt_definitions().items():
        payload = {**definition, "commitMessage": "Migrate existing FactorMiner prompt to management"}
        body_file = args.output_dir / f"{observation}.json"
        body_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if not args.apply:
            print(f"Prepared {definition['name']}: {body_file}")
            continue
        status, existing = request_cli(["get", definition["name"], "--label", "production"], env)
        if status == 200 and same_template(existing, definition):
            print(f"Unchanged {definition['name']} v{existing['version']}")
            continue
        _, published = request_cli(["create", "--body-file", str(body_file.resolve())], env)
        print(f"Published {definition['name']} v{published['version']} (production)")


if __name__ == "__main__":
    main()
