"""The CLI's .env loader: syntax, precedence, and no-op when absent."""

from __future__ import annotations

import os

from factorminer.utils.env_file import load_env_file


def test_loads_unset_keys_with_quotes_comments_and_export(tmp_path, monkeypatch) -> None:
    for key in ("FM_T_A", "FM_T_B", "FM_T_C", "FM_T_D", "FM_T_EMPTY"):
        monkeypatch.delenv(key, raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "# comment\n"
        "FM_T_A=plain-value  # trailing comment\n"
        'export FM_T_B="quoted # not a comment"\n'
        "FM_T_C='single'\n"
        "FM_T_EMPTY=\n"
        "not a valid line\n",
        encoding="utf-8",
    )
    loaded = load_env_file(env)
    assert sorted(loaded) == ["FM_T_A", "FM_T_B", "FM_T_C"]
    assert os.environ["FM_T_A"] == "plain-value"
    assert os.environ["FM_T_B"] == "quoted # not a comment"
    assert os.environ["FM_T_C"] == "single"
    assert "FM_T_EMPTY" not in os.environ


def test_shell_environment_wins_over_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FM_T_KEY", "from-shell")
    env = tmp_path / ".env"
    env.write_text("FM_T_KEY=from-file\n", encoding="utf-8")
    assert load_env_file(env) == []
    assert os.environ["FM_T_KEY"] == "from-shell"


def test_missing_file_and_env_override_path(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FACTORMINER_ENV_FILE", raising=False)
    assert load_env_file() == []
    custom = tmp_path / "keys.env"
    custom.write_text("FM_T_CUSTOM=1\n", encoding="utf-8")
    monkeypatch.delenv("FM_T_CUSTOM", raising=False)
    monkeypatch.setenv("FACTORMINER_ENV_FILE", str(custom))
    assert load_env_file() == ["FM_T_CUSTOM"]
