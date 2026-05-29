"""Standalone backtest: REGIME-ADAPTIVE MOMENTUM (regime_momentum_n500).

Offline research/validation path. The SELECTION + REGIME + RISK logic lives in
the shared core `strategy.py` (rank_targets / regime_healthy / hit_stop /
hit_trail / params) — the SAME code live_signal.py uses, so backtest and live
cannot drift. This file only owns the offline walk: load panels, build the
monthly calendar, replay buy/sell, score.

Reproduces (true N500, net 0.15%/side):
  FULL 2023-05..2026-05 : +69.1% CAGR / 27.4% DD / Calmar 2.5
    per-year 2023 +112, 2024 +77, 2025 +23, 2026 +3.8 (partial bear)
  WINDOW Mar-2025..May-2026 : +46.8% total / +38% annualized / 20.3% DD

Run: python3 tools/models/regime_momentum_n500/backtest.py [--start 2023-05-15] [--end 2026-05-12]
"""
import sys, json, argparse
from pathlib import Path
from datetime import date, datetime, timedelta

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import pandas as pd
from sqlalchemy import text
from tools.shared.ohlcv_cache import _get_engine
from tools.shared.universes import nifty500_symbols
from tools.models.regime_momentum_n500 import strategy as S

COST = 0.0015
DEFAULT_START = date(2023, 5, 15)
DEFAULT_END = date(2026, 5, 12)
DEFAULT_CAP = 1_000_000.0


def load_panels(eng, start, end):
    syms = [f"NSE:{s}-EQ" for s, _ in nifty500_symbols()] + [S.INDEX]
    with eng.connect() as c:
        df = pd.read_sql(text(
            "SELECT symbol,date,close,volume FROM historical_data "
            "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b AND data_source='fyers' "
            "ORDER BY symbol,date"
        ), c, params={"s": syms, "a": start - timedelta(days=420), "b": end})
    df["date"] = pd.to_datetime(df["date"])
    df["adv"] = df["close"].astype(float) * df["volume"].astype(float)
    cl = df.pivot(index="date", columns="symbol", values="close").astype(float)
    adv_rs = df.pivot(index="date", columns="symbol", values="adv")
    # Restrict to EQUITY trading days — index-only date rows would poison the
    # rolling ADV/return windows (same guard as live_signal).
    equity_dates = adv_rs.drop(columns=[S.INDEX], errors="ignore").dropna(how="all").index
    cl = cl.loc[equity_dates].ffill()
    adv_rs = adv_rs.loc[equity_dates]
    adv20, idx_sma200 = S.indicators(cl, adv_rs)
    return cl, adv20, idx_sma200


def month_starts(dates):
    s = pd.Series(dates, index=dates)
    return [pd.Timestamp(x) for x in s.groupby([dates.year, dates.month]).first().values]


def _px(cl, sym, d):
    try:
        v = cl.at[d, sym]
        return float(v) if pd.notna(v) else 0.0
    except KeyError:
        return 0.0


def run(start, end, capital, out_dir=None):
    eng = _get_engine()
    cl, adv20, idx_sma200 = load_panels(eng, start, end)
    dates = cl.index
    rebal = set(d for d in month_starts(dates) if start <= d.date() <= end)
    cash = capital
    pos = {}                     # sym -> {qty, entry_px, entry_date, peak}
    nav, trades = [], []

    def sell(s, d, reason):
        nonlocal cash
        px = _px(cl, s, d)
        if px <= 0:
            return
        p = pos[s]
        cash += p["qty"] * px * (1 - COST)
        trades.append({"sym": s.replace("NSE:", "").replace("-EQ", ""),
                       "entry_date": p["entry_date"].date().isoformat(),
                       "exit_date": d.date().isoformat(),
                       "ret_pct": round((px / p["entry_px"] - 1) * 100, 2),
                       "reason": reason})
        del pos[s]

    for d in dates:
        if d.date() < start or d.date() > end:
            continue
        di = dates.get_loc(d)
        healthy = S.regime_healthy(cl, idx_sma200, di)
        # DAILY bear-regime risk exits
        for s in list(pos.keys()):
            px = _px(cl, s, d)
            if px <= 0:
                continue
            p = pos[s]; p["peak"] = max(p["peak"], px)
            if S.hit_trail(px, p["peak"], healthy):
                sell(s, d, "TRAIL"); continue
            if S.hit_stop(px, p["entry_px"], healthy):
                sell(s, d, "STOP"); continue
        nav.append((d, cash + sum(p["qty"] * _px(cl, s, d) for s, p in pos.items())))
        if d not in rebal:
            continue
        ranked = S.rank_targets(cl, adv20, di)
        if not ranked:
            for s in list(pos.keys()):
                sell(s, d, "CASH")
            continue
        keep = set(ranked[:S.RETAIN])
        for s in list(pos.keys()):
            if s not in keep:
                sell(s, d, "ROTATE")
        want = [s for s in ranked if s not in pos][: max(0, S.K - len(pos))]
        if want:
            per = cash / len(want)
            for s in want:
                px = _px(cl, s, d)
                if px <= 0:
                    continue
                q = int(per / (px * (1 + COST)))
                if q >= 1:
                    cash -= q * px * (1 + COST)
                    pos[s] = {"qty": q, "entry_px": px, "entry_date": d, "peak": px}

    navs = pd.Series({d: v for d, v in nav}).dropna()
    final = navs.iloc[-1]
    yrs = (end - start).days / 365.25
    cagr = (final / capital) ** (1 / yrs) - 1
    dd = ((navs.cummax() - navs) / navs.cummax()).max()
    py = {int(y): round((g.iloc[-1] / g.iloc[0] - 1) * 100, 1)
          for y, g in navs.groupby(navs.index.year)}
    wins = sum(1 for t in trades if t["ret_pct"] > 0)
    result = {
        "model": "regime_momentum_n500",
        "start": start.isoformat(), "end": end.isoformat(),
        "final_nav": round(final, 0), "cagr_pct": round(cagr * 100, 1),
        "max_dd_pct": round(dd * 100, 1), "calmar": round(cagr * 100 / max(0.01, dd * 100), 2),
        "trades": len(trades), "win_rate_pct": round(wins / max(1, len(trades)) * 100, 1),
        "per_year": py,
    }
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "summary.json").write_text(json.dumps(result, indent=2))
        (Path(out_dir) / "trade_ledger.json").write_text(json.dumps(trades, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=DEFAULT_START.isoformat())
    ap.add_argument("--end", default=DEFAULT_END.isoformat())
    ap.add_argument("--capital", type=float, default=DEFAULT_CAP)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    s = datetime.strptime(a.start, "%Y-%m-%d").date()
    e = datetime.strptime(a.end, "%Y-%m-%d").date()
    r = run(s, e, a.capital, a.out_dir)
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
