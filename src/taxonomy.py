"""Tiny helper around config/flaw_types.yaml (the single source of truth).

Every component should use this instead of re-parsing or re-defining the taxonomy.
Units: seconds, dB, semitones. Severity levels are 1..5 (0 = not flagged).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config" / "flaw_types.yaml"


@lru_cache(maxsize=None)
def load_taxonomy(path: str | None = None) -> dict[str, Any]:
    with open(path or DEFAULT_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def flaw_ids() -> list[str]:
    return list(load_taxonomy()["flaws"].keys())


def get_flaw(flaw_id: str) -> dict[str, Any]:
    return load_taxonomy()["flaws"][flaw_id]


def severity_label(level: int) -> str:
    return next(l["id"] for l in load_taxonomy()["levels"] if l["level"] == level)


def level_for_metric(flaw_id: str, metric_value: float) -> int:
    """Map a detection metric value (see flaw.detection.metric) to severity 1..5; 0 if below threshold."""
    bounds = get_flaw(flaw_id)["detection"]["severity_bounds"]
    level = 0
    for i, b in enumerate(bounds, start=1):
        if metric_value >= b:
            level = i
    return level


def render_summary(flaw_id: str, **fields: Any) -> str:
    """Fill the flaw's explanation summary template with measured values."""
    return get_flaw(flaw_id)["explanation"]["summary_template"].format(**fields)
