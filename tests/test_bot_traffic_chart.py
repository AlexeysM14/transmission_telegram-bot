from __future__ import annotations

import builtins
from datetime import datetime
from typing import Any

import bot


def test_traffic_charts_have_png_fallback_without_matplotlib(monkeypatch: Any) -> None:
    original_import = builtins.__import__

    def import_without_matplotlib(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "matplotlib":
            raise ImportError("matplotlib is unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_matplotlib)
    points = [
        {"date": "01.08", "downloaded": 2 * 1024**3, "uploaded": 1024**3},
        {"date": "02.08", "downloaded": 3 * 1024**3, "uploaded": 2 * 1024**3},
    ]

    weekly_payload, weekly_error = bot._build_traffic_chart_last_7_days(points)
    assert weekly_error is None
    assert weekly_payload is not None and weekly_payload.startswith(b"\x89PNG\r\n\x1a\n")

    history = [
        {"date": "2026-08-01", "downloaded": 0, "uploaded": 0},
        {"date": "2026-08-02", "downloaded": 2 * 1024**3, "uploaded": 1024**3},
    ]
    day_points, monthly_payload, monthly_error = bot._build_traffic_chart_current_month(
        datetime(2026, 8, 2, 12, 0),
        3 * 1024**3,
        2 * 1024**3,
        history,
    )
    assert day_points is not None
    assert monthly_error is None
    assert monthly_payload is not None and monthly_payload.startswith(b"\x89PNG\r\n\x1a\n")
