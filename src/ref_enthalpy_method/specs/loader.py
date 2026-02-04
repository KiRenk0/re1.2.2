"""YAML loading helpers (compatible with ref_enthalpy/specs schema)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class SpecError(RuntimeError):
    pass


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise SpecError(f"Spec not found: {p}") from e
    except Exception as e:
        raise SpecError(f"Failed to read YAML: {p} ({e})") from e

    if not isinstance(data, dict):
        raise SpecError(f"Invalid YAML root (expected mapping): {p}")
    return data

