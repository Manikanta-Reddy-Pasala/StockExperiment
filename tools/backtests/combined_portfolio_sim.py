"""Combined 3-bucket portfolio sim under cross-model overlap policies.

The three large-cap momentum models (n100, pseudo, n20) trade ONE shared Fyers
account, each with its own capital bucket. They frequently pick the SAME stock
(58 collision events / 3yr — see analyze_model_overlap.py), which (a) corrupts
per-model ledgers via the net-merging reconciler and (b) collapses
diversification by concentrating multiple buckets into one name.

This sim quantifies the RETURN/RISK cost of three policies:
  allow : each bucket independently buys its rank-1 (current live behaviour;
          equals the sum of the isolated per-model backtests). Overlap allowed.
  block : sequential priority order; a bucket whose rank-1 is already held by a
          higher-priority bucket sits in CASH that cycle.
  rank2 : same priority; a blocked bucket walks down its own ranked list to the
          first name no sibling currently holds.

Dedup gates NEW entries only — it never force-sells a retained winner. midcap
is excluded by construction (its universe excludes Nifty 100 -> never collides).

Each model reproduces its OWN backtest selection EXACTLY (same universe,
lookback, filters, calendar) so the `allow` result reconciles with the
published per-model summaries. Execution mirrors the shared engine
(decide_rotation, same-day close fills, no fees).

Usage (must run where historical_data is reachable, e.g. inside the app
container on the VM):
    python tools/backtests/combined_portfolio_sim.py \
        --from 2023-05-15 --to 2026-05-12 --capital 1000000
"""
import sys, csv, json, argparse
from pathlib import Path
from datetime import date, timedelta

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pandas as pd
from sqlalchemy import text
from tools.shared.ohlcv_cache import _get_engine
from tools.shared.universes import nifty500_symbols
from tools.shared.rotation_strategy import decide_rotation

N100_CSV = str(ROOT / "src" / "data" / "symbols" / "nifty100.csv")
SML_CSV = str(ROOT / "src" / "data" / "symbols" / "nifty_smallcap250.csv")

# Priority order for sequential dedup: flagship real-N100 first, then pseudo,
# then the high-churn daily n20 last (it rebalances every day, easiest to give
# the leftover slot).
PRIORITY = ["n100", "pseudo", "n20"]


def _load_csv_syms(path):
    out = set()
    try:
        with open(path) as f:
            for r in csv.DictReader(f):
                if r.get("Series", "").strip() == "EQ":
                    out.add(r["Symbol"].strip())
    except FileNotFoundError:
        pass
    return out


def _label(s):
    return s.replace("NSE:", "").replace("-EQ", "")


def monthly_calendar(start, end, dates):
    """First trading day on/after the 1st of each month in [start,end] + start."""
    out = set()
    y, m = start.year, start.month
    while True:
        fut = dates[dates >= pd.Timestamp(y, m, 1)]
        if len(fut) == 0 or fut[0].date() > end:
            break
        if fut[0].date() >= start:
            out.add(fut[0])
        m += 1
        if m > 12:
            m = 1; y += 1
    sd = pd.Timestamp(start)
    if sd in dates:
        out.add(sd)
    return out


class Bucket:
    """One model's capital bucket: cash + a single optional long position."""

    def __init__(self, name, capital, rank_at, rebal_dates, lookback):
        self.name = name
        self.cap = capital
        self.rank_at = rank_at          # di -> ranked symbols (best first)
        self.rebal = rebal_dates        # set of pd.Timestamp rebalance days
        self.lookback = lookback
        self.hold = None
        self.qty = 0
        self.entry_px = 0.0
        self.entry_date = None
        self.trades = []

    def mtm(self, close, di):
        if self.hold and pd.notna(close[self.hold].iloc[di]):
            return self.cap + self.qty * float(close[self.hold].iloc[di])
        return self.cap

    def _sell(self, close, d, di, reason):
        px = close[self.hold].iloc[di]
        if pd.isna(px):
            return
        px = float(px)
        proc = self.qty * px
        self.cap += proc
        self.trades.append({
            "sym": _label(self.hold), "entry_date": self.entry_date,
            "exit_date": d.date().isoformat(), "qty": self.qty,
            "entry_px": round(self.entry_px, 2), "exit_px": round(px, 2),
            "pnl": round(proc - self.qty * self.entry_px, 0),
            "ret_pct": round((px / self.entry_px - 1) * 100, 2),
            "exit_reason": reason,
        })
        self.hold = None; self.qty = 0

    def _buy(self, close, d, di, sym):
        px = close[sym].iloc[di]
        if pd.isna(px):
            return
        px = float(px)
        q = int(self.cap / px)
        if q >= 1 and q * px <= self.cap:
            self.cap -= q * px
            self.hold = sym; self.qty = q; self.entry_px = px
            self.entry_date = d.date().isoformat()


def build_models(cl, adv20, sma200, dates, start, end, n100_set, smallcap):
    """Return {name: Bucket-config} reproducing each model's exact selection."""
    present_n100 = [f"NSE:{s}-EQ" for s in n100_set
                    if f"NSE:{s}-EQ" in cl.columns]

    # ---- n100: real NSE Nifty 100, 15d return, no filters, monthly ----
    def n100_rank(di, _lb=15):
        if di < _lb:
            return []
        univ = [s for s in present_n100 if pd.notna(cl[s].iloc[di])]
        rets = cl.iloc[di].reindex(univ) / cl.iloc[di - _lb].reindex(univ) - 1
        return list(rets.dropna().sort_values(ascending=False).index)

    # ---- pseudo: yearly-PIT top-100 ADV (N500) minus smallcap, 30d, SMA200+₹3000 ----
    year_starts = []
    cur = start
    while cur <= end:
        year_starts.append(pd.Timestamp(cur)); cur = cur.replace(year=cur.year + 1)
    year_univ = {}
    for ys in year_starts:
        fut = dates[dates >= ys]
        if len(fut) == 0:
            continue
        di = dates.get_loc(fut[0])
        top = adv20.iloc[di].dropna().sort_values(ascending=False).head(100).index.tolist()
        year_univ[ys] = [s for s in top if _label(s) not in smallcap]

    def pseudo_univ(d):
        chosen = year_starts[0]
        for ys in year_starts:
            if d >= ys:
                chosen = ys
        return year_univ.get(chosen, [])

    def pseudo_rank(di, _lb=30):
        if di < max(_lb, 200):
            return []
        univ = pseudo_univ(dates[di])
        up = sma200.iloc[di] < cl.iloc[di]
        univ = [s for s in univ if bool(up.get(s, False))]
        univ = [s for s in univ
                if pd.notna(cl[s].iloc[di]) and float(cl[s].iloc[di]) <= 3000]
        if not univ:
            return []
        rets = cl.iloc[di].reindex(univ) / cl.iloc[di - _lb].reindex(univ) - 1
        return list(rets.dropna().sort_values(ascending=False).index)

    # ---- n20: daily top-20 ADV ∩ Nifty100, 30d, SMA200 ----
    def n20_rank(di, _lb=30):
        if di < max(_lb, 200):
            return []
        pit = adv20.iloc[di].dropna().sort_values(ascending=False).head(20).index.tolist()
        up = sma200.iloc[di] < cl.iloc[di]
        pit = [s for s in pit if bool(up.get(s, False))]
        pit = [s for s in pit if _label(s) in n100_set]
        if not pit:
            return []
        rets = cl.iloc[di].reindex(pit) / cl.iloc[di - _lb].reindex(pit) - 1
        return list(rets.dropna().sort_values(ascending=False).index)

    monthly = monthly_calendar(start, end, dates)
    daily = {d for d in dates if start <= d.date() <= end}
    return {
        "n100": (n100_rank, monthly, 15),
        "pseudo": (pseudo_rank, monthly, 30),
        "n20": (n20_rank, daily, 30),
    }


def run_policy(policy, configs, cl, dates, capital, start, end):
    """Run all 3 buckets over the daily timeline under `policy`. Returns dict."""
    buckets = {n: Bucket(n, capital, *configs[n]) for n in PRIORITY}
    trading = [d for d in dates if start <= d.date() <= end]
    nav_series = []

    for d in trading:
        di = dates.get_loc(d)
        # Process buckets in priority order so a lower-priority bucket sees the
        # holds higher-priority ones just (re)entered this same day.
        for name in PRIORITY:
            b = buckets[name]
            if d not in b.rebal:
                continue
            ranked = b.rank_at(di)
            if not ranked:
                continue
            dec = decide_rotation(b.hold, ranked, 1)
            if dec.is_noop:
                continue  # retain — dedup never force-exits a kept winner
            others = {buckets[o].hold for o in PRIORITY
                      if o != name and buckets[o].hold}
            if policy == "allow":
                target = ranked[0]
            elif policy == "block":
                target = ranked[0] if ranked[0] not in others else None
            elif policy == "rank2":
                target = next((s for s in ranked if s not in others), None)
            else:
                raise ValueError(policy)
            if target == b.hold:
                continue  # landed back on current hold — nothing to do
            if b.hold:
                b._sell(cl, d, di, "ROTATE")
            if target:
                b._buy(cl, d, di, target)
        # Mark combined NAV after the day's rebalances.
        nav_series.append(sum(b.mtm(cl, di) for b in buckets.values()))

    nav = pd.Series(nav_series)
    final = float(nav.iloc[-1])
    total_cap = capital * len(PRIORITY)
    yrs = (end - start).days / 365.25
    cagr = ((final / total_cap) ** (1 / yrs) - 1) * 100 if yrs > 0 else 0.0
    roll = nav.cummax()
    dd = float(((roll - nav) / roll).max()) * 100 if len(nav) > 1 else 0.0
    per = {n: {"final": round(b.mtm(cl, len(dates) - 1), 0),
               "trades": len(b.trades),
               "hold": _label(b.hold) if b.hold else None}
           for n, b in buckets.items()}
    return {
        "policy": policy,
        "combined_final": round(final, 0),
        "combined_return_pct": round((final / total_cap - 1) * 100, 2),
        "combined_cagr_pct": round(cagr, 2),
        "combined_max_dd_pct": round(dd, 2),
        "calmar": round(cagr / max(0.01, dd), 2),
        "total_trades": sum(len(b.trades) for b in buckets.values()),
        "per_bucket": per,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="2023-05-15")
    ap.add_argument("--to", dest="end", default="2026-05-12")
    ap.add_argument("--capital", type=float, default=1_000_000.0,
                    help="Per-bucket starting capital (3 buckets total).")
    ap.add_argument("--data-source", default="fyers")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    start = date.fromisoformat(a.start); end = date.fromisoformat(a.end)

    eng = _get_engine()
    n500 = [f"NSE:{s}-EQ" for s, _ in nifty500_symbols()]
    with eng.connect() as c:
        df = pd.read_sql(text(
            "SELECT symbol,date,close,volume FROM historical_data "
            "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b AND data_source=:ds "
            "ORDER BY symbol,date"
        ), c, params={"s": n500, "a": start - timedelta(days=400), "b": end,
                      "ds": a.data_source})
    df["date"] = pd.to_datetime(df["date"])
    df["adv_rs"] = df["close"].astype(float) * df["volume"].astype(float)
    cl = df.pivot(index="date", columns="symbol", values="close").ffill()
    adv_rs = df.pivot(index="date", columns="symbol", values="adv_rs").fillna(0)
    adv20 = adv_rs.rolling(20).mean()
    sma200 = cl.rolling(200).mean()
    dates = cl.index
    print(f"Loaded {len(dates)} days × {len(cl.columns)} symbols")

    n100_set = _load_csv_syms(N100_CSV)
    smallcap = _load_csv_syms(SML_CSV)
    configs = build_models(cl, adv20, sma200, dates, start, end, n100_set, smallcap)

    results = [run_policy(p, configs, cl, dates, a.capital, start, end)
               for p in ("allow", "block", "rank2")]

    print(f"\n{'='*78}\nCOMBINED 3-BUCKET PORTFOLIO  (₹{a.capital:,.0f} each, "
          f"₹{a.capital*3:,.0f} total)  {start} → {end}\n{'='*78}")
    hdr = f"{'policy':8s} {'CAGR%':>8s} {'MaxDD%':>8s} {'Calmar':>7s} {'final₹':>14s} {'trades':>7s}"
    print(hdr); print("-" * len(hdr))
    for r in results:
        print(f"{r['policy']:8s} {r['combined_cagr_pct']:>8.2f} "
              f"{r['combined_max_dd_pct']:>8.2f} {r['calmar']:>7.2f} "
              f"{r['combined_final']:>14,.0f} {r['total_trades']:>7d}")
    print("\nPer-bucket final NAV + current hold:")
    for r in results:
        print(f"  [{r['policy']}] " + " | ".join(
            f"{n}: ₹{v['final']:,.0f} ({v['trades']}t, hold={v['hold']})"
            for n, v in r["per_bucket"].items()))

    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
