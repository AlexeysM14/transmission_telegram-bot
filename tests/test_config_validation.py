from __future__ import annotations

import math

import pytest

import bot


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0", "-1"])
def test_positive_float_environment_rejects_non_finite_and_non_positive_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("TEST_TIMEOUT", value)

    with pytest.raises(RuntimeError, match="finite number"):
        bot._parse_float_env("TEST_TIMEOUT", "10")


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_boolean_environment_accepts_explicit_true_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("TEST_BOOLEAN", value)

    assert bot._parse_bool_env("TEST_BOOLEAN") is True


@pytest.mark.parametrize("value", ["0", "false", "NO", "off"])
def test_boolean_environment_accepts_explicit_false_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("TEST_BOOLEAN", value)

    assert bot._parse_bool_env("TEST_BOOLEAN", default=True) is False


def test_boolean_environment_rejects_typos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOW_ALL_USERS", "treu")

    with pytest.raises(RuntimeError, match="ALLOW_ALL_USERS must be one of"):
        bot._parse_bool_env("ALLOW_ALL_USERS")


@pytest.mark.parametrize(
    "value",
    [
        "http://",
        "http://:8080",
        "http://proxy host:8080",
        "http://127.0.0.1:99999",
        "socks5://[invalid",
        "ftp://127.0.0.1:21",
    ],
)
def test_proxy_url_validation_rejects_malformed_urls(value: str) -> None:
    with pytest.raises(RuntimeError):
        bot._normalize_proxy_url(value, env_name="TG_PROXY")


def test_proxy_url_masking_is_fail_safe_and_removes_secrets() -> None:
    assert bot._mask_proxy_url("http://user:secret@[invalid") == "<invalid URL>"
    masked = bot._mask_proxy_url("http://user:secret@127.0.0.1:9091/rpc?token=hidden#fragment")
    assert masked == "http://***:***@127.0.0.1:9091/rpc"
    assert "secret" not in masked
    assert "hidden" not in masked


@pytest.mark.parametrize(
    "value",
    [
        "file:///etc/passwd",
        "http://",
        "http://transmission host:9091/rpc",
        "http://127.0.0.1:99999/rpc",
        "http://127.0.0.1:9091/rpc?session=1",
        "http://127.0.0.1:9091/rpc#fragment",
    ],
)
def test_transmission_url_validation_rejects_malformed_urls(value: str) -> None:
    with pytest.raises(RuntimeError):
        bot._validate_transmission_url(value)


def test_load_config_validates_transmission_host_path_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "TG_PROXY",
        "TG_GET_UPDATES_PROXY",
        "TR_URL",
        "TR_PROTOCOL",
        "TR_PORT",
        "LIST_LIMIT",
        "ALLOWED_USER_IDS",
        "ALLOW_ALL_USERS",
        "BOT_TIMEZONE",
        "STATE_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TG_TOKEN", "test-token")
    monkeypatch.setenv("TR_HOST", "https://127.0.0.1")

    with pytest.raises(RuntimeError, match="TR_HOST"):
        bot.load_config()

    monkeypatch.setenv("TR_HOST", "127.0.0.1")
    monkeypatch.setenv("TR_PATH", "transmission/rpc")
    with pytest.raises(RuntimeError, match="TR_PATH"):
        bot.load_config()

    monkeypatch.setenv("TR_PATH", "/transmission/rpc")
    monkeypatch.setenv("TR_TIMEOUT", "nan")
    with pytest.raises(RuntimeError, match="finite number"):
        bot.load_config()

    monkeypatch.setenv("TR_TIMEOUT", "5.5")
    config = bot.load_config()
    assert math.isclose(config.tr_timeout, 5.5)


def test_transmission_url_ignores_stale_fallback_endpoint_values(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "TG_PROXY",
        "TG_GET_UPDATES_PROXY",
        "TR_TIMEOUT",
        "LIST_LIMIT",
        "ALLOWED_USER_IDS",
        "ALLOW_ALL_USERS",
        "BOT_TIMEZONE",
        "STATE_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TG_TOKEN", "test-token")
    monkeypatch.setenv("TR_URL", "https://transmission.example:9443/transmission/rpc")
    monkeypatch.setenv("TR_PROTOCOL", "invalid")
    monkeypatch.setenv("TR_HOST", "https://stale.invalid")
    monkeypatch.setenv("TR_PORT", "not-a-port")
    monkeypatch.setenv("TR_PATH", "stale/relative/path")

    config = bot.load_config()

    assert config.tr_url == "https://transmission.example:9443/transmission/rpc"
    assert (config.tr_protocol, config.tr_host, config.tr_port, config.tr_path) == (
        "http",
        "127.0.0.1",
        9091,
        "/transmission/rpc",
    )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_torrent_numeric_formatting_handles_non_finite_rpc_values(value: float) -> None:
    assert bot._non_negative_int(value) is None
    assert bot._non_negative_float(value) is None
    assert bot._clamp_progress(value) == 0.0
    assert bot.fmt_bytes(value) == "0 B"
