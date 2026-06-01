"""Lock the shared from-entry hard-stop helpers (pseudo ATR / n100 fixed-%)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.shared.stops import atr_stop_level, atr_stop_hit, fixed_stop_level, fixed_stop_hit


def test_atr_level():
    assert atr_stop_level(100.0, 5.0, 3.0) == 85.0          # 100 - 3*5

def test_atr_level_bad_inputs():
    assert atr_stop_level(100, None, 3.0) is None
    assert atr_stop_level(0, 5, 3) is None
    assert atr_stop_level(100, 5, 0) is None
    assert atr_stop_level(10, 5, 3) is None                 # level <=0 -> None

def test_atr_hit():
    hit, lvl = atr_stop_hit(100.0, 5.0, 84.0, 3.0)          # low 84 <= 85
    assert hit is True and lvl == 85.0

def test_atr_not_hit():
    hit, lvl = atr_stop_hit(100.0, 5.0, 86.0, 3.0)          # low 86 > 85
    assert hit is False and lvl == 85.0

def test_fixed_level():
    assert fixed_stop_level(100.0, 0.12) == 88.0            # -12%

def test_fixed_hit():
    hit, lvl = fixed_stop_hit(100.0, 87.5, 0.12)            # 87.5 <= 88
    assert hit is True and lvl == 88.0

def test_fixed_not_hit():
    hit, lvl = fixed_stop_hit(100.0, 90.0, 0.12)
    assert hit is False and lvl == 88.0

def test_fixed_bad_inputs():
    assert fixed_stop_level(None, 0.12) is None
    assert fixed_stop_hit(100, None, 0.12) == (False, 88.0)
