from __future__ import annotations

import logging

import bot


def test_event_formatter_redacts_telegram_token_and_url_credentials(monkeypatch) -> None:
    telegram_token = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"
    transmission_password = "transmission-secret"
    monkeypatch.setattr(bot, "_LOG_SECRETS", (transmission_password,))

    record = logging.LogRecord(
        name="httpx",
        level=logging.INFO,
        pathname=__file__,
        lineno=12,
        msg=(
            "HTTP Request: POST "
            f"https://api.telegram.org/bot{telegram_token}/sendMessage "
            f"via http://rpc-user:{transmission_password}@transmission.local:9091/transmission/rpc"
        ),
        args=(),
        exc_info=None,
    )
    formatter = bot.EventTimeFormatter("%(levelname)s | %(name)s | %(message)s")

    output = formatter.format(record)

    assert telegram_token not in output
    assert transmission_password not in output
    assert "rpc-user" not in output
    assert "/bot<redacted>/sendMessage" in output
    assert "http://***:***@transmission.local" in output


def test_redaction_covers_secrets_outside_urls(monkeypatch) -> None:
    monkeypatch.setattr(bot, "_LOG_SECRETS", ("proxy-password", "rpc-password"))

    output = bot._redact_log_text("proxy-password and rpc-password must never be logged")

    assert output == "<redacted> and <redacted> must never be logged"
