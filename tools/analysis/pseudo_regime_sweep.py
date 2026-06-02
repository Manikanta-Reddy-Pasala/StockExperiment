"""pseudo_n100_adv — INDEX-REGIME + momentum-floor + uptrend sweep (research only).

User idea (2026-06-02): stop entering blindly on momentum rank. Add three gates:
  1. Momentum FLOOR — don't enter unless the rank-1 30d return >= floor.
  2. Stock UPTREND — only names trading above their 200d SMA (S.SMA_GATE).
  3. INDEX REGIME — read NIFTY50: when the index is ABOVE its SMA (uptrend) enter
     normally; when BELOW (downtrend) DON'T enter unless the candidate's momentum
     is WAY higher (a much bigger down-floor). Optional: also EXIT to cash while
     the index is in a downtrend (sit out bear phases → cut drawdown).

Selection replicates the production pseudo backtest EXACTLY (yearly fixed-anchor
top-100-ADV PIT N500, drop smallcap, MAX_PRICE<=3000, rank by 30d return,
rank-1 rotation, from-entry ATR(14) k=3 hard stop). Index = NSE:NIFTY50-INDEX
(history from 2023-01), so the comparison window is 2023→2026.

Run:
  python3 tools/analysis/pseudo_regime_sweep.py --from 2023-02-01 --to 2026-05-20
"""
import sys, argparse
from pathlib import Path
from datetime import date, datetime, timedelta

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
from sqlalchemy import text

from tools.shared.ohlcv_cache import _get_engine
from tools.shared.rotation_strategy import decide_rotation
from tools.models.momentum_pseudo_n100_adv import strategy as S
from tools.models.momentum_pseudo_n100_adv.strategy import build_calendar
from tools.analysis.pseudo_improve_sweep import (
    load_panels, atr_panel, build_universes, make_rank_at, _metrics,
)

INDEX_SYM = "NSE:NIFTY50-INDEX"


def load_index(eng, start, end, data_source="fyers"):
    with eng.connect() as c:
        df = pd.read_sql(text(
            "SELECT date, close FROM historical_data "
            "WHERE symbol=:s AND date BETWEEN :a AND :b AND data_source=:ds ORDER BY date"
        ), c, params={"s": INDEX_SYM, "a": start - timedelta(days=400), "b": end, "ds": data_source})
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"].astype(float)


def regime_sim(dates, cl, hi, lo, atr, calendar, rank_at_base, *, capital, start, end,
               idx, idx_sma, lb, floor_up, floor_down, regime_exit):
    """Daily-MTM single-position walk with the production ATR-k3 stop, PLUS the
    regime + momentum-floor entry gate. regime_exit=True also liquidates to cash
    while the index is in a downtrend."""
    cal = {pd.Timestamp(d): kind for d, kind in calendar}
    cash = capital; nav_marks = [capital]; nav_by_day = []
    hold = None; qty = 0; entry_px = 0.0; entry_date = None
    ATR_K = S.ATR_STOP_MULT

    # align index + its sma onto the trading-day grid
    idx_a = idx.reindex(dates).ffill()
    idx_sma_a = idx_sma.reindex(dates).ffill()

    def index_up(di):
        a, b = idx_a.iloc[di], idx_sma_a.iloc[di]
        if pd.isna(a) or pd.isna(b):
            return True   # unknown regime → treat as up (don't over-block)
        return a >= b

    def gated_rank(di):
        r = rank_at_base(di)
        if not r:
            return []
        up = index_up(di)
        floor = floor_up if up else floor_down
        out = []
        for s in r:
            try:
                ret = float(cl[s].iloc[di] / cl[s].iloc[di - lb] - 1) * 100
            except Exception:
                continue
            if ret >= floor:
                out.append(s)
        return out

    first_di = dates.get_loc(min(cal)) if cal else 0
    cur = None
    for di in range(first_di, len(dates)):
        d = dates[di]
        px = float(cl[hold].iloc[di]) if hold and pd.notna(cl[hold].iloc[di]) else None
        nav_marks.append(cash + (qty * px if hold and px else 0.0))
        nav_by_day.append((d, nav_marks[-1]))

        # daily ATR-from-entry hard stop (production rule)
        if hold and qty > 0 and ATR_K and entry_px > 0:
            a = atr[hold].iloc[di] if hold in atr.columns else None
            if a is not None and pd.notna(a) and a > 0:
                lvl = entry_px - ATR_K * float(a)
                low = float(lo[hold].iloc[di]) if pd.notna(lo[hold].iloc[di]) else px
                if lvl > 0 and low is not None and low <= lvl:
                    cash += qty * lvl
                    hold = None; qty = 0; entry_px = 0.0

        # regime exit: liquidate while index is in a downtrend (optional)
        if regime_exit and hold and qty > 0 and not index_up(di) and px:
            cash += qty * px
            hold = None; qty = 0; entry_px = 0.0

        kind = cal.get(d)
        if kind is None:
            continue
        ranked = gated_rank(di)
        if not ranked:
            continue
        top = ranked[0]
        if decide_rotation(hold, ranked, retain_top_n=1).is_noop:
            continue
        if hold and qty > 0:
            sx = float(cl[hold].iloc[di]); cash += qty * sx; hold = None; qty = 0
        bx = float(cl[top].iloc[di])
        if bx > 0:
            q = int(cash / bx)
            if q >= 1 and q * bx <= cash:
                cash -= q * bx; qty = q; hold = top; entry_px = bx
                entry_date = d.date().isoformat()

    final = cash + (qty * float(cl[hold].iloc[-1]) if hold else 0.0)
    return _metrics(final, capital, [], nav_marks, start, end, nav_by_day)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="2023-02-01")
    ap.add_argument("--to", dest="end", default="2026-05-20")
    ap.add_argument("--capital", type=float, default=100000.0)
    a = ap.parse_args()
    start = date.fromisoformat(a.start); end = date.fromisoformat(a.end)

    eng = _get_engine()
    cl, hi, lo, adv20, sma200, sma20 = load_panels(eng, start, end)
    atr = atr_panel(cl, hi, lo)
    dates = cl.index
    year_starts, yu = build_universes(dates, adv20, start, end)
    rank_at = make_rank_at(dates, cl, sma200, year_starts, yu)
    cal = build_calendar(dates, start, end, mid_check=False)
    idx = load_index(eng, start, end)

    def run(name, **kw):
        idx_sma = idx.rolling(kw.pop("idx_sma_win")).mean()
        m = regime_sim(dates, cl, hi, lo, atr, cal, rank_at, capital=a.capital,
                       start=start, end=end, idx=idx, idx_sma=idx_sma, lb=S.LOOKBACK, **kw)
        calmar = (m['cagr'] / m['dd']) if m['dd'] else 0.0
        print(f"  {name:<42} CAGR {m['cagr']:+7.1f}%  DD {m['dd']:5.1f}%  Calmar {calmar:5.2f}  ret {m['ret']:+.0f}%")

    print(f"=== pseudo regime sweep  {start} → {end}  (index=NIFTY50, ATR-k{S.ATR_STOP_MULT} stop on) ===")
    print(f"  SMA_GATE(stock uptrend)={S.SMA_GATE}")
    # baseline = current production (no floor, no regime): floor_up=-999, floor_down=-999, no regime_exit
    run("baseline (production, no regime)", idx_sma_win=50, floor_up=-9999, floor_down=-9999, regime_exit=False)
    for win in (50, 100, 200):
        run(f"regime entry-gate (idxSMA{win}, up>=0 down>=10)", idx_sma_win=win, floor_up=0, floor_down=10, regime_exit=False)
        run(f"regime entry-gate (idxSMA{win}, up>=0 down>=15)", idx_sma_win=win, floor_up=0, floor_down=15, regime_exit=False)
        run(f"regime EXIT+gate (idxSMA{win}, up>=0 down>=15)",  idx_sma_win=win, floor_up=0, floor_down=15, regime_exit=True)
        run(f"regime EXIT+gate (idxSMA{win}, up>=5 down>=20)",  idx_sma_win=win, floor_up=5, floor_down=20, regime_exit=True)
    # pure momentum floor (no regime) for comparison
    run("momentum floor up>=5 (no regime)", idx_sma_win=50, floor_up=5, floor_down=5, regime_exit=False)
    run("momentum floor up>=10 (no regime)", idx_sma_win=50, floor_up=10, floor_down=10, regime_exit=False)


if __name__ == "__main__":
    main()
