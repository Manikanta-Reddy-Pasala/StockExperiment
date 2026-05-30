"""Cross-model holding-overlap analysis.

Each large-cap momentum model (n100, pseudo, n20) holds max 1 symbol at a
time, all trading on ONE shared Fyers account. When two models hold the SAME
symbol over overlapping dates, the broker merges them into a single net
holding the per-model reconciler can't attribute -> ledger corruption +
capital concentration. midcap is excluded (Nifty100-excluded universe -> can
never overlap with the other three).

This reads each model's backtest trade_ledger.json (+ its open position) and
reports: how often models overlap on the same symbol, total overlap days, and
the specific (symbol, date-range) collisions. Pure JSON crunch, no DB.

Usage:
    python tools/backtests/analyze_model_overlap.py \
        --dir exports/overlap --end 2026-05-12
"""
import sys, json, argparse
from pathlib import Path
from datetime import date
from itertools import combinations

# Models that share the large-cap Nifty100 universe (can collide). midcap is
# intentionally absent — its universe excludes Nifty100 by construction.
LARGE_CAP_MODELS = [
    "momentum_n100_top5_max1",
    "momentum_pseudo_n100_adv",
    "n20_daily_large_only",
]


def load_holdings(model_dir: Path, model: str, end: date):
    """Return list of (sym, entry_date, exit_date) intervals for one model.

    Closed trades come from trade_ledger.json; the still-open position (if any)
    is read from summary.json and capped at `end` so it counts as held through
    the window end.
    """
    holds = []
    led = model_dir / f"{model}.json"
    if not led.exists():
        led = model_dir / model / "trade_ledger.json"
    trades = json.loads(led.read_text())
    for t in trades:
        if t.get("entry_date") and t.get("exit_date"):
            holds.append((t["sym"],
                          date.fromisoformat(t["entry_date"]),
                          date.fromisoformat(t["exit_date"])))
    # Open position(s) held through window end — single open_position OR multi
    # open_positions (multi-holding models like momentum_retest_n500).
    summ = model_dir / model / "summary.json"
    if summ.exists():
        sj = json.loads(summ.read_text())
        ops = sj.get("open_positions") or (
            [sj["open_position"]] if sj.get("open_position") else [])
        for op in ops:
            if op.get("sym") and op.get("entry_date"):
                holds.append((op["sym"], date.fromisoformat(op["entry_date"]), end))
    return holds


def overlap_days(a, b):
    """Inclusive day-count where intervals a=(s,e), b=(s,e) overlap; 0 if none."""
    lo = max(a[1], b[1])
    hi = min(a[2], b[2])
    return (hi - lo).days + 1 if hi >= lo else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="exports/overlap",
                    help="Dir holding <model>.json ledgers (and <model>/summary.json)")
    ap.add_argument("--end", default="2026-05-12", help="Window end (caps open pos)")
    a = ap.parse_args()
    base = Path(a.dir)
    end = date.fromisoformat(a.end)

    per_model = {m: load_holdings(base, m, end) for m in LARGE_CAP_MODELS}
    for m, h in per_model.items():
        print(f"{m}: {len(h)} holding intervals")

    print("\n===== PAIRWISE SYMBOL-OVERLAP =====")
    grand_days = 0
    grand_events = 0
    for m1, m2 in combinations(LARGE_CAP_MODELS, 2):
        pair_days = 0
        events = []
        for h1 in per_model[m1]:
            for h2 in per_model[m2]:
                if h1[0] != h2[0]:
                    continue  # different symbol — no concentration collision
                d = overlap_days(h1, h2)
                if d > 0:
                    lo = max(h1[1], h2[1]); hi = min(h1[2], h2[2])
                    events.append((h1[0], lo, hi, d))
                    pair_days += d
        grand_days += pair_days
        grand_events += len(events)
        tag = f"{m1.split('_')[0]} x {m2.split('_')[1] if m2.startswith('momentum') else m2.split('_')[0]}"
        print(f"\n-- {m1}  ⨯  {m2} --")
        if not events:
            print("   NO overlap ever")
        else:
            print(f"   {len(events)} collision events, {pair_days} overlap-days total:")
            for sym, lo, hi, d in sorted(events, key=lambda x: x[1]):
                print(f"     {sym:14s} {lo} → {hi}  ({d}d)")

    print(f"\n===== TOTAL: {grand_events} collision events, "
          f"{grand_days} overlap-days across the window =====")


if __name__ == "__main__":
    main()
