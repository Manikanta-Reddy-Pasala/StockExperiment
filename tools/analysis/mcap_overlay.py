"""A/B test: does layering the NSE free-float MARKET-CAP signal on top of an
existing model's selection lift CAGR / win-rate / Calmar?

How (zero model-code duplication, zero drift): each single-position model runs
its OWN backtest.run() unchanged, but we monkeypatch the `run_rotation_backtest`
symbol IN THAT MODULE's namespace with a wrapper that intercepts the model's
`rank_at` closure and post-filters/re-orders its output with the mcap overlay.
Baseline and every overlay use the identical engine + calendar + universe, so
any delta is purely the overlay.

Overlays (vs BASELINE = the model's own ranking):
  band   : keep only names whose reconstructed FF-mcap rank is in [LO, HI]
  climber: keep names whose FF-mcap rank IMPROVED over the last CLIMB_LOOKBACK
           trading days (rising toward index inclusion), best climbers first
  blend  : order by (momentum_rank + W * mcap_rank) — mcap as a tiebreaker

FF-mcap panel: ff_shares = current FF-mcap / latest DB close; ffmcap[t] =
ff_shares * close[t]; daily descending rank. PIT on price, current shares.

Works for the single-position rotation models (emerging / n100 / pseudo / n20).
retest (multi-holding K=3) + midcap (event-driven) use different engines —
tested separately.

Run: python3 tools/analysis/mcap_overlay.py --model <name>
     python3 tools/analysis/mcap_overlay.py --all
"""
from __future__ import annotations
import sys, csv, argparse, warnings, importlib
from pathlib import Path
from datetime import date
import numpy as np, pandas as pd
from sqlalchemy import text
warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from tools.shared.ohlcv_cache import _get_engine

MCAP_CSV = ROOT / "exports" / "nse_mcap.csv"
FULL = (date(2023, 5, 15), date(2026, 5, 12))
CLIMB_LOOKBACK = 60
BLEND_W = 0.5

# model -> (backtest module path, run kwargs, mcap-rank band appropriate to its
# universe size). emerging/midcap-ish = mid/small (80-600); large-cap = 20-160.
MODELS = {
    "emerging_momentum":        ("tools.models.emerging_momentum.backtest",        {}, (80, 600)),
    "momentum_n100_top5_max1":  ("tools.models.momentum_n100_top5_max1.backtest",  {}, (20, 160)),
    "momentum_pseudo_n100_adv": ("tools.models.momentum_pseudo_n100_adv.backtest", {}, (20, 300)),
    "n20_daily_large_only":     ("tools.models.n20_daily_large_only.backtest",     {}, (10, 120)),
}


def load_ffmcap():
    out = {}
    for r in csv.DictReader(open(MCAP_CSV)):
        try:
            ff = float(r["ff_mcap_cr"])
            if ff > 0:
                out[f"NSE:{r['symbol']}-EQ"] = ff
        except (ValueError, TypeError):
            continue
    return out


def build_mcap_rank(cl, ff_mcap):
    """date-indexed descending FF-mcap rank panel (1 = biggest)."""
    ff_shares = {}
    for s in ff_mcap:
        if s in cl.columns:
            last = cl[s].dropna()
            if len(last) and last.iloc[-1] > 0:
                ff_shares[s] = ff_mcap[s] * 1e7 / last.iloc[-1]
    eq = list(ff_shares)
    ffmcap = cl[eq].mul(pd.Series(ff_shares), axis=1)
    return ffmcap.rank(axis=1, ascending=False, method="first")


def make_overlay_patch(orig_engine, overlay, rank, band, climb_lb, holder):
    """Drop-in run_rotation_backtest: wraps rank_at with the overlay + captures
    the engine's result dataclass into `holder` (run()'s own return varies)."""
    def mrank(s, dt):
        try:
            v = rank.at[dt, s]
            return float(v) if pd.notna(v) else np.nan
        except KeyError:
            return np.nan

    def patched(*a, **kw):
        dates = kw["dates"]; base_rank_at = kw["rank_at"]

        def wrapped(di):
            base = base_rank_at(di)
            if not overlay or not base:
                return base
            dt = dates[di]
            if overlay == "band":
                return [s for s in base if not np.isnan(mrank(s, dt)) and band[0] <= mrank(s, dt) <= band[1]]
            if overlay == "climber":
                dt0 = dates[max(0, di - climb_lb)]
                return [s for s in base if not np.isnan(mrank(s, dt)) and not np.isnan(mrank(s, dt0))
                        and mrank(s, dt) < mrank(s, dt0)]
            if overlay == "blend":
                mr = {s: i for i, s in enumerate(base)}
                return sorted(base, key=lambda s: mr[s]
                              + BLEND_W * (mrank(s, dt) if not np.isnan(mrank(s, dt)) else 9999))
            return base

        kw["rank_at"] = wrapped
        res = orig_engine(*a, **kw)
        holder["res"] = res
        return res
    return patched


def run_model(model_name, overlay, rank):
    modpath, run_kw, band = MODELS[model_name]
    mod = importlib.import_module(modpath)
    orig = mod.run_rotation_backtest
    holder = {}
    mod.run_rotation_backtest = make_overlay_patch(orig, overlay, rank, band, CLIMB_LOOKBACK, holder)
    try:
        mod.run(FULL[0], FULL[1], 1_000_000.0, **run_kw)
    finally:
        mod.run_rotation_backtest = orig
    return holder["res"]


def fmt(tag, r, base=None):
    cagr = r.cagr_pct; dd = r.max_dd_pct; cal = r.calmar
    wr = r.wins / max(1, r.wins + r.losses) * 100
    delta = f"  (Δ {cagr - base:+.1f}pp)" if base is not None else ""
    print(f"  {tag:9s}: CAGR {cagr:+6.1f}% | DD {dd:5.1f}% | Calmar {cal:4.2f} | "
          f"trades {len(r.trades):3d} | WR {wr:4.1f}%{delta}")
    return cagr


def run_one(model_name, rank):
    _, _, band = MODELS[model_name]
    print(f"\n## {model_name} — mcap overlay A/B  band[{band[0]},{band[1]}] "
          f"climber(lb{CLIMB_LOOKBACK}) blend(w{BLEND_W})")
    base = None
    for tag, ov in (("BASELINE", None), ("band", "band"), ("climber", "climber"), ("blend", "blend")):
        try:
            r = run_model(model_name, ov, rank)
            c = fmt(tag, r, None if tag == "BASELINE" else base)
            if tag == "BASELINE":
                base = c
        except Exception as e:
            print(f"  {tag}: ERROR {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="emerging_momentum")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()

    ff_mcap = load_ffmcap()
    print(f"FF-mcap names: {len(ff_mcap)}")
    from tools.models.emerging_momentum import backtest as B
    cl, _ = B.load_panels(_get_engine(), FULL[0], FULL[1])
    rank = build_mcap_rank(cl, ff_mcap)

    models = list(MODELS) if a.all else [a.model]
    for m in models:
        run_one(m, rank)


if __name__ == "__main__":
    main()
