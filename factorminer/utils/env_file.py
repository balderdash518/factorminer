"""Load API keys from a local ``.env`` file into the process environment.

Lookup: ``$FACTORMINER_ENV_FILE`` if set, otherwise ``.env`` in the working
directory. Variables already present in the environment win, so a shell
``export`` always overrides the file. Values are never logged.

Supported syntax: ``KEY=value``, optional ``export`` prefix, ``#`` comments,
and single- or double-quoted values.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


def _parse_value(raw: str) -> str:
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
        return raw[1:-1]
    # Unquoted: strip an inline comment introduced by whitespace + '#'.
    return re.split(r"\s+#", raw, maxsplit=1)[0].strip()


def load_env_file(path: str | Path | None = None) -> list[str]:
    """Set unset variables from the env file; return the names that were set."""
    if path is None:
        path = os.environ.get("FACTORMINER_ENV_FILE") or ".env"
    env_path = Path(path)
    if not env_path.is_file():
        return []
    loaded: list[str] = []
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _LINE.match(line)
        if match is None:
            continue
        key, value = match.group(1), _parse_value(match.group(2))
        if key in os.environ or not value:
            continue
        os.environ[key] = value
        loaded.append(key)
    if loaded:
        logger.debug("Loaded %d variable(s) from %s: %s", len(loaded), env_path, ", ".join(loaded))
    return loaded
