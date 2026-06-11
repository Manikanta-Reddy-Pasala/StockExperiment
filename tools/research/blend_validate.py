"""Validate the emerging-2x tilt BLEND with CURRENT model configs (post PT=0).

Method (proven 2026-06-05 blend_opt): inject a one-line daily-NAV csv dump into
each model's backtest.py right before run()'s return, run the real backtest
(same code path as live), RESTORE the file in finally, then blend the 5 daily
NAV series: equal-weight and emerging-2x tilt (fixed initial weights, weighted
sum of normalized NAVs — same convention as the 06-07 study).

Run inside the app container:
  python tools/research/blend_validate.py --from 2021-03-01 --to 2026-06-10
"""
import argparse, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pandas as pd

# model dir -> (nav var name, the run()-return line to inject before)
MODELS = {
    "emerging_momentum":         ("nser", "    return result"),
    "momentum_pseudo_n100_adv":  ("_nav", "    return final, cagr, trades"),
    "momentum_n100_top5_max1":   ("_nav", "    return final, cagr, trades"),
    "n40":                       ("_nav", "    return final, cagr, trades"),
    "momentum_retest_n500":      ("nav",  "    return final, cagr, trades"),
}


def run_one(mdir, navvar, ret_line, start, end, cap):
    bt = ROOT / "tools" / "models" / mdir / "backtest.py"
    orig = bt.read_text()
    out_csv = f"/tmp/nav_{mdir}.csv"
    dump = f"    {navvar}.to_csv('{out_csv}')\n"
    assert ret_line + "\n" in orig, f"{mdir}: return line not found"
    bt.write_text(orig.replace(ret_line + "\n", dump + ret_line + "\n", 1))
    try:
        r = subprocess.run(
            [sys.executable, str(bt), "--from", start, "--to", end,
             "--capital", str(cap)],
            capture_output=True, text=True, timeout=1800, cwd=str(ROOT))
        tail = (r.stdout + r.stderr).strip().splitlines()[-8:]
        print(f"== {mdir} (exit {r.returncode}) ==")
        for ln in tail:
            print("  " + ln)
        if r.returncode != 0:
            return None
    finally:
        bt.write_text(orig)        # ALWAYS restore the production file
    s = pd.read_csv(out_csv, index_col=0)
    ser = s.iloc[:, 0]
    ser.index = pd.to_datetime(ser.index)
    return ser.sort_index()


def metrics(nav, name):
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = ((nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1) * 100
    roll = nav.cummax(); mdd = float(((roll - nav) / roll).max()) * 100
    print(f"{name:28s} CAGR {cagr:+7.1f}%  MaxDD {mdd:5.1f}%  "
          f"Calmar {cagr / max(0.5, mdd):5.2f}")
    for yy, g in nav.groupby(nav.index.year):
        if len(g) < 2:
            continue
        rl = g.cummax(); dd = float(((rl - g) / rl).max()) * 100
        print(f"    {yy}: {(g.iloc[-1]/g.iloc[0]-1)*100:+7.1f}%  (DD {dd:.1f}%)")
    return cagr, mdd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="2021-03-01")
    ap.add_argument("--to", dest="end", default="2026-06-10")
    ap.add_argument("--capital", type=float, default=100000)
    ap.add_argument("--skip-run", action="store_true",
                    help="reuse existing /tmp/nav_*.csv dumps")
    a = ap.parse_args()

    navs = {}
    for mdir, (navvar, ret_line) in MODELS.items():
        if a.skip_run:
            s = pd.read_csv(f"/tmp/nav_{mdir}.csv", index_col=0).iloc[:, 0]
            s.index = pd.to_datetime(s.index)
            navs[mdir] = s.sort_index()
        else:
            ser = run_one(mdir, navvar, ret_line, a.start, a.end, a.capital)
            if ser is None:
                print(f"!! {mdir} FAILED — aborting"); return
            navs[mdir] = ser

    # align on common dates, normalize to 1.0
    df = pd.DataFrame(navs).dropna()
    df = df / df.iloc[0]
    print(f"\nblend panel: {len(df)} common days "
          f"{df.index[0].date()}..{df.index[-1].date()}\n")
    for m in df.columns:
        metrics(df[m], m)
    print()
    eq = df.mean(axis=1)
    metrics(eq, "BLEND equal-weight")
    w = pd.Series(1.0, index=df.columns); w["emerging_momentum"] = 2.0
    w = w / w.sum()
    tilt = (df * w).sum(axis=1)
    print()
    metrics(tilt, "BLEND emerging-2x tilt")


if __name__ == "__main__":
    main()
