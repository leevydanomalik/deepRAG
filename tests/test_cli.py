from unittest.mock import patch

from typer.testing import CliRunner

from app.cli import app


def test_cli_patterns_subcommand():
    runner = CliRunner()
    with patch("app.cli.list_patterns", return_value=["naive", "agentic", "graph", "loop"]):
        r = runner.invoke(app, ["patterns"])
    assert r.exit_code == 0
    assert "naive" in r.stdout


def test_cli_ask_invokes_registry():
    runner = CliRunner()
    with patch("app.cli.run", return_value={"answer": "ok", "trace": [], "raw": {}}) as m:
        r = runner.invoke(app, ["ask", "naive", "hello?"])
    assert r.exit_code == 0
    assert "ok" in r.stdout
    m.assert_called_once_with("naive", "hello?")
