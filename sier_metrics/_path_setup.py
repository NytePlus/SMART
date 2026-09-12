"""Ensure imports work when running scripts from scripts/26.8.2/."""

from __future__ import annotations

import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
TESTING_SCRIPTS = WORKSPACE_ROOT / "benchmark" / "testing" / "scripts"
SCRIPTS_DIR = Path(__file__).resolve().parent

for path in (SCRIPTS_DIR, TESTING_SCRIPTS):
    entry = str(path)
    if entry not in sys.path:
        sys.path.insert(0, entry)
