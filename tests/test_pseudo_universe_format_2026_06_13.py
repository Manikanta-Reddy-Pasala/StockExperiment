"""pseudo yearly-universe format contract (2026-06-13 latent-crash fix).

Bug: data_pull.refresh_universe wrote the yearly universe key as a list of
{"symbol": s} dicts while live_signal consumes entries as plain strings
(`sym in _SMALLCAP_SET`, f"NSE:{sym}-EQ"). The on-disk 2023/24/25 keys are
strings (live healthy), but the NEXT yearly rebuild (2027-05-01) would have
written dicts and crashed every signal run with
TypeError: unhashable type 'dict'.

Contract locked here:
  1. The merge writer (_merge_snapshot_into_universes) ALWAYS writes plain
     symbol strings, whatever shape the build snapshot has.
  2. load_yearly_universes normalizes BOTH formats (str and {"symbol": s})
     so any already-written dict-format file self-heals on load.

Pure file I/O via tmp_path — no DB, no Fyers. Run:
    python3 -m pytest tests/test_pseudo_universe_format_2026_06_13.py -q
"""
import json

from tools.models.momentum_pseudo_n100_adv.data_pull import (
    _merge_snapshot_into_universes,
)
from tools.models.momentum_pseudo_n100_adv.live_signal import (
    load_yearly_universes,
    pick_universe_for,
)
from datetime import datetime


SYMS = [f"SYM{i:03d}" for i in range(100)]  # >=50 passes the sanity floor


# ---------------------------------------------------------------------------
# Writer: refresh_universe's merge step must emit plain strings.
# ---------------------------------------------------------------------------

def _write_snapshot(tmp_path, payload):
    p = tmp_path / "snapshot.json"
    p.write_text(json.dumps(payload))
    return str(p)


def test_merge_writes_plain_strings_from_dict_snapshot(tmp_path):
    # build_universe snapshot shape: {"stocks": [{"symbol": ...}, ...]}
    snap = _write_snapshot(tmp_path, {"stocks": [{"symbol": s} for s in SYMS]})
    uni = tmp_path / "yearly_universes.json"
    assert _merge_snapshot_into_universes(snap, str(uni), "2027-05-01") is True
    written = json.loads(uni.read_text())
    assert list(written.keys()) == ["2027-05-01"]
    entries = written["2027-05-01"]
    assert all(isinstance(e, str) for e in entries), \
        "yearly universe entries MUST be plain strings (live_signal hashes them)"
    assert entries == SYMS


def test_merge_writes_plain_strings_from_bare_list_snapshot(tmp_path):
    snap = _write_snapshot(tmp_path, SYMS)  # bare list of strings
    uni = tmp_path / "yearly_universes.json"
    assert _merge_snapshot_into_universes(snap, str(uni), "2027-05-01") is True
    entries = json.loads(uni.read_text())["2027-05-01"]
    assert all(isinstance(e, str) for e in entries)
    assert entries == SYMS


def test_merge_preserves_prior_year_keys(tmp_path):
    uni = tmp_path / "yearly_universes.json"
    uni.write_text(json.dumps({"2025-05-13": ["OLDSYM1", "OLDSYM2"]}))
    snap = _write_snapshot(tmp_path, {"symbols": SYMS})
    assert _merge_snapshot_into_universes(snap, str(uni), "2027-05-01") is True
    written = json.loads(uni.read_text())
    assert written["2025-05-13"] == ["OLDSYM1", "OLDSYM2"]  # untouched
    assert written["2027-05-01"] == SYMS


def test_merge_skips_partial_build_below_sanity_floor(tmp_path):
    uni = tmp_path / "yearly_universes.json"
    uni.write_text(json.dumps({"2025-05-13": ["KEEP"]}))
    snap = _write_snapshot(tmp_path, SYMS[:10])  # <50 = bad/partial build
    assert _merge_snapshot_into_universes(snap, str(uni), "2027-05-01") is False
    assert json.loads(uni.read_text()) == {"2025-05-13": ["KEEP"]}  # not clobbered


# ---------------------------------------------------------------------------
# Loader: must self-heal a dict-format file AND pass strings through.
# ---------------------------------------------------------------------------

def test_loader_normalizes_dict_entries(tmp_path):
    # Simulate a file written by the OLD buggy refresh_universe.
    p = tmp_path / "yearly_universes.json"
    p.write_text(json.dumps({
        "2025-05-13": ["HFCL", "IDEA"],                 # healthy string format
        "2027-05-01": [{"symbol": s} for s in SYMS],    # buggy dict format
    }))
    yearly = load_yearly_universes(str(p))
    assert yearly["2025-05-13"] == ["HFCL", "IDEA"]
    assert yearly["2027-05-01"] == SYMS
    for entries in yearly.values():
        assert all(isinstance(e, str) for e in entries)


def test_loader_output_is_hashable_and_pick_universe_works(tmp_path):
    # The exact crash path: rank_universe does `plain_sym in set` and
    # f"NSE:{plain_sym}-EQ". Dict entries raise TypeError on the set probe.
    p = tmp_path / "yearly_universes.json"
    p.write_text(json.dumps({"2027-05-01": [{"symbol": s} for s in SYMS]}))
    yearly = load_yearly_universes(str(p))
    key, symbols = pick_universe_for(datetime(2027, 6, 1), yearly)
    assert key == "2027-05-01"
    smallcap_set = {"SYM001", "NOTTHERE"}
    assert any(sym in smallcap_set for sym in symbols)  # hashable: no TypeError
    assert [f"NSE:{sym}-EQ" for sym in symbols[:2]] == \
        ["NSE:SYM000-EQ", "NSE:SYM001-EQ"]


def test_loader_handles_empty_and_null_year_values(tmp_path):
    p = tmp_path / "yearly_universes.json"
    p.write_text(json.dumps({"2025-05-13": [], "2026-05-01": None}))
    yearly = load_yearly_universes(str(p))
    assert yearly == {"2025-05-13": [], "2026-05-01": []}


def test_real_universe_file_on_disk_is_strings():
    # Guard the PRODUCTION artifact itself: every entry in the checked-in
    # yearly_universes.json must already be a plain string.
    import tools.models.momentum_pseudo_n100_adv.live_signal as LS
    from pathlib import Path
    real = Path(LS.__file__).parent / "yearly_universes.json"
    raw = json.loads(real.read_text())
    for key, entries in raw.items():
        assert all(isinstance(e, str) for e in entries), \
            f"{key} contains non-string entries — run loader self-heal/rewrite"
