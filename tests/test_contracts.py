import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_contracts_validate():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_contracts.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_taxonomy_helper():
    sys.path.insert(0, str(ROOT))
    from src import taxonomy as t

    assert len(t.flaw_ids()) == 6
    assert t.level_for_metric("pace_fast", 5) == 0
    assert t.level_for_metric("pace_fast", 35) == 3
    assert t.severity_label(5) == "severe"
