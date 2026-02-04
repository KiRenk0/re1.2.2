"""Lightweight warning collection + optional file logging (baseline-like)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class WarningLog:
    path: Path
    enabled: bool = True
    warnings: list[str] = field(default_factory=list)

    def reset_file(self) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(f"REM Solver Warnings - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")

    def warn(self, msg: str) -> None:
        self.warnings.append(str(msg))
        if not self.enabled:
            return
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%H:%M:%S')} | {msg}\n")
        except Exception:
            # Logging failure should not kill a run.
            return

