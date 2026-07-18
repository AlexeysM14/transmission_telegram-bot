from __future__ import annotations

import importlib.machinery
import importlib.util
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_cli_module() -> ModuleType:
    path = ROOT / "transmission3-bot"
    loader = importlib.machinery.SourceFileLoader("transmission3_bot_cli", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise RuntimeError("Failed to create CLI module spec")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cli() -> ModuleType:
    return load_cli_module()


@pytest.mark.parametrize("value", ["NaN", "inf", "-inf", "0", "-1", "not-a-number"])
def test_parse_timeout_rejects_non_finite_and_non_positive_values(cli: ModuleType, value: str) -> None:
    timeout = cli.parse_timeout(value)  # type: ignore[attr-defined]

    assert math.isfinite(timeout)
    assert timeout == 10.0


@pytest.mark.parametrize(
    "value",
    [
        "http://127.0.0.1:99999",
        "socks5://[invalid",
        "http://",
        "http://proxy host:8080",
        "ftp://127.0.0.1:21",
    ],
)
def test_proxy_validation_rejects_malformed_urls(cli: ModuleType, value: str) -> None:
    with pytest.raises(ValueError):
        cli.normalize_proxy_url_or_raise(value, env_name="TG_PROXY")  # type: ignore[attr-defined]


def test_status_url_masking_is_fail_safe_and_removes_secrets(cli: ModuleType) -> None:
    assert cli.mask_proxy_url("http://user:secret@[invalid") == "<invalid URL>"  # type: ignore[attr-defined]
    masked = cli.mask_proxy_url("http://user:secret@127.0.0.1:9091/rpc?token=hidden#fragment")  # type: ignore[attr-defined]
    assert masked == "http://***:***@127.0.0.1:9091/rpc"
    assert "secret" not in masked
    assert "hidden" not in masked


def test_status_url_validation_and_tcp_check_fail_cleanly(cli: ModuleType) -> None:
    with pytest.raises(ValueError):
        cli.validate_http_url("file:///etc/passwd", label="TR_URL")  # type: ignore[attr-defined]
    with pytest.raises(ValueError):
        cli.validate_http_url("http://127.0.0.1:9091/rpc?unsafe=1", label="TR_URL")  # type: ignore[attr-defined]

    ok, message = cli.check_tcp_target(  # type: ignore[attr-defined]
        "http://127.0.0.1:99999",
        timeout=1.0,
        label="proxy",
    )
    assert ok is False
    assert "invalid URL" in message


def test_subprocess_environment_removes_shell_proxy_and_ca_overrides(
    cli: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dangerous = {
        "BASH_ENV": "/tmp/execute-me",
        "ENV": "/tmp/execute-me-too",
        "BASHOPTS": "extglob",
        "HTTP_PROXY": "http://attacker.invalid",
        "https_proxy": "http://attacker.invalid",
        "SSL_CERT_FILE": "/tmp/untrusted-ca.pem",
        "SSLKEYLOGFILE": "/tmp/tls-keys.log",
        "REQUESTS_CA_BUNDLE": "/tmp/untrusted-ca.pem",
    }
    for key, value in dangerous.items():
        monkeypatch.setenv(key, value)

    sanitized = cli.sanitized_subprocess_env()  # type: ignore[attr-defined]

    assert dangerous.keys().isdisjoint(sanitized)
    assert sanitized["PATH"] == "/usr/sbin:/usr/bin:/sbin:/bin"


def test_command_failure_uses_redacted_display_command(
    cli: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli.subprocess,  # type: ignore[attr-defined]
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1),
    )

    with pytest.raises(RuntimeError) as exc_info:
        cli.run(  # type: ignore[attr-defined]
            ["git", "clone", "https://user:secret@example.invalid/repo.git"],
            display_cmd=["git", "clone", "<configured repository>"],
        )

    message = str(exc_info.value)
    assert "secret" not in message
    assert "<configured repository>" in message


@pytest.mark.parametrize(
    ("returncode", "stdout"),
    [
        (1, ""),
        (0, "not-a-commit"),
        (0, "a" * 39),
    ],
)
def test_checkout_commit_must_be_a_verified_object_id(
    cli: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    stdout: str,
) -> None:
    monkeypatch.setattr(cli, "trusted_system_executable", lambda _: "/usr/bin/git")
    monkeypatch.setattr(
        cli.subprocess,  # type: ignore[attr-defined]
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=returncode, stdout=stdout),
    )

    with pytest.raises(RuntimeError, match="clean checkout commit"):
        cli.resolve_checkout_commit(Path("/trusted/checkout"))  # type: ignore[attr-defined]


def test_checkout_commit_is_normalized_before_deployment(cli: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "trusted_system_executable", lambda _: "/usr/bin/git")
    monkeypatch.setattr(
        cli.subprocess,  # type: ignore[attr-defined]
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=("A" * 40) + "\n"),
    )

    assert cli.resolve_checkout_commit(Path("/trusted/checkout")) == "a" * 40  # type: ignore[attr-defined]


def test_socks_proxy_uses_venv_health_check(
    cli: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, float, str]] = []

    def fake_check(token: str, *, proxy_url: str, timeout: float, label: str) -> tuple[bool, str]:
        assert token == "telegram-token"
        calls.append((proxy_url, timeout, label))
        return True, f"{label}: ok"

    monkeypatch.setattr(cli, "check_telegram_api_via_socks", fake_check)

    result = cli.check_telegram_api(  # type: ignore[attr-defined]
        "telegram-token",
        proxy_url="socks5://127.0.0.1:1080",
        timeout=3.0,
        label="Telegram",
    )

    assert result == (True, "Telegram: ok")
    assert calls == [("socks5://127.0.0.1:1080", 3.0, "Telegram")]
