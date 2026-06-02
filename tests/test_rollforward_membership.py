"""tools/analysis/rollforward_membership — roll the PIT membership table's
leading edge forward to the current NSE list without the factsheet PDFs.

Covers: no-op when in sync, closing a leaver, opening a joiner, alias no-churn
(renamed name is not seen as leave+join), and the sanity guard that refuses to
mutate history on a suspiciously small current list.
"""
import csv
from pathlib import Path

import tools.analysis.rollforward_membership as RF

SENT = "2099-12-31"


def _write_csv(path: Path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def _setup(tmp_path, monkeypatch, current_syms, membership_rows):
    monkeypatch.setattr(RF, "SYMBOLS_DIR", tmp_path)
    _write_csv(tmp_path / "nifty100.csv",
               ["Company Name", "Industry", "Symbol", "Series", "ISIN Code"],
               [["x", "y", s, "EQ", "z"] for s in current_syms])
    _write_csv(tmp_path / "n100_membership.csv",
               ["symbol", "start_date", "end_date"], membership_rows)


def _open_members(tmp_path):
    out = set()
    with (tmp_path / "n100_membership.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            if r["end_date"] == SENT:
                out.add(r["symbol"])
    return out


def test_noop_when_in_sync(tmp_path, monkeypatch):
    cur = [f"SYM{i}" for i in range(95)]
    mem = [[s, "2024-01-01", SENT] for s in cur]
    _setup(tmp_path, monkeypatch, cur, mem)
    nl, nj, aborted = RF.rollforward_index("n100", "nifty100.csv", 90, "2026-06-02", False)
    assert (nl, nj, aborted) == (0, 0, False)


def test_closes_leaver_and_opens_joiner(tmp_path, monkeypatch):
    base = [f"SYM{i}" for i in range(94)]
    # OLDNAME is open in the table but NOT in the current list -> must close.
    # NEWNAME is in the current list but not open -> must open.
    cur = base + ["NEWNAME"]
    mem = [[s, "2024-01-01", SENT] for s in base] + [["OLDNAME", "2024-01-01", SENT]]
    _setup(tmp_path, monkeypatch, cur, mem)
    nl, nj, aborted = RF.rollforward_index("n100", "nifty100.csv", 90, "2026-06-02", False)
    assert aborted is False and nl == 1 and nj == 1
    opens = _open_members(tmp_path)
    assert "OLDNAME" not in opens and "NEWNAME" in opens
    # OLDNAME interval closed at as_of, history preserved (row still present).
    with (tmp_path / "n100_membership.csv").open(newline="") as fh:
        rows = {r["symbol"]: r for r in csv.DictReader(fh)}
    assert rows["OLDNAME"]["end_date"] == "2026-06-02"
    assert rows["NEWNAME"]["start_date"] == "2026-06-02"
    # Backup written.
    assert (tmp_path / "n100_membership.csv.rollforward.bak").exists()


def test_alias_rename_no_churn(tmp_path, monkeypatch):
    # Table still stores the OLD symbol (ZOMATO); current list has the NEW one
    # (ETERNAL). _TICKER_ALIAS maps ZOMATO->ETERNAL, so this must be a no-op,
    # NOT ZOMATO-left + ETERNAL-joined.
    assert RF._TICKER_ALIAS.get("ZOMATO") == "ETERNAL"
    base = [f"SYM{i}" for i in range(94)]
    cur = base + ["ETERNAL"]
    mem = [[s, "2024-01-01", SENT] for s in base] + [["ZOMATO", "2024-01-01", SENT]]
    _setup(tmp_path, monkeypatch, cur, mem)
    nl, nj, aborted = RF.rollforward_index("n100", "nifty100.csv", 90, "2026-06-02", False)
    assert (nl, nj, aborted) == (0, 0, False)


def test_aborts_on_suspiciously_small_current(tmp_path, monkeypatch):
    # A partial / failed download (only 10 names) must NOT truncate history.
    cur = [f"SYM{i}" for i in range(10)]
    mem = [[f"SYM{i}", "2024-01-01", SENT] for i in range(95)]
    _setup(tmp_path, monkeypatch, cur, mem)
    nl, nj, aborted = RF.rollforward_index("n100", "nifty100.csv", 90, "2026-06-02", False)
    assert aborted is True
    # Nothing closed.
    assert len(_open_members(tmp_path)) == 95


def test_dry_run_does_not_write(tmp_path, monkeypatch):
    base = [f"SYM{i}" for i in range(94)]
    cur = base + ["NEWNAME"]
    mem = [[s, "2024-01-01", SENT] for s in base]
    _setup(tmp_path, monkeypatch, cur, mem)
    nl, nj, aborted = RF.rollforward_index("n100", "nifty100.csv", 90, "2026-06-02", True)
    assert nj == 1 and aborted is False
    assert "NEWNAME" not in _open_members(tmp_path)  # unchanged on disk
    assert not (tmp_path / "n100_membership.csv.rollforward.bak").exists()
