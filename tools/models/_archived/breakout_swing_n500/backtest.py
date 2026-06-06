"""breakout_swing_n500 — short-hold (1-5d) momentum-burst swing backtest (numpy).

Two entry alphas, both short-hold, both exited by the SHARED
``breakout_exit_reason`` core (TARGET/STOP/TRAIL/MAX_HOLD on TRADING-day age):

  - "breakout" : fresh 40d-high + >200d SMA + volume surge (is_breakout core),
                 ranked by volume-surge ratio. (original spec)
  - "pullback" : Medium-article idea — uptrend (>200d SMA) name that has pulled
                 back to just above its 20-EMA (dip-buy), with positive lookback
                 momentum, ranked by closeness to the EMA (deepest healthy dip).

NO LOOKAHEAD: every entry/exit decision uses the OBSERVED close at index ``di``
and transacts at that same close. Ranking input == transaction price. (The ORB
sin was ranking ``di`` then trading ``di``'s open.)

Speed: all panels are numpy arrays; config-independent masks (breakout base,
price-range, vol-ratio, pullback base) are precomputed ONCE so each sweep config
only does cheap per-day boolean ANDs + a tiny per-position exit loop.

CLI:
  python tools/models/breakout_swing_n500/backtest.py            # defaults, 1 run
  python tools/models/breakout_swing_n500/backtest.py --sweep    # full grid
  python tools/models/breakout_swing_n500/backtest.py --mode pullback
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from datetime import date, timedelta

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from sqlalchemy import text
from tools.shared.ohlcv_cache import _get_engine
from tools.shared.breakout_strategy import breakout_exit_reason
from tools.shared.index_membership import eligible_at, universe_union
from tools.models.breakout_swing_n500 import strategy as S

DEFAULT_START = date(2021, 3, 1)
DEFAULT_END   = date(2026, 5, 29)
DEFAULT_CAP   = 1_000_000.0
SLIP, BR, STT = 0.001, 20, 0.001

# pullback band: close must sit between +1% and +8% above the 20-EMA (a dip that
# held above support, not a breakdown), in an uptrend, with positive momentum.
PB_EMA      = 20
PB_LO, PB_HI = 0.0, 0.08    # 0–8% above EMA20
PB_MOM_FLOOR = 5.0          # require >= +5% LOOKBACK-day return (still in an upmove)
PB_LOOKBACK = 20


@dataclass
class BacktestConfig:
    slots: int        = S.SLOTS
    target_pct: float = S.TARGET_PCT
    stop_pct: float   = S.STOP_PCT
    trail_pct: float  = S.TRAIL_PCT
    profit_trig: float = S.PROFIT_TRIG
    max_hold: int     = S.MAX_HOLD
    vol_mult: float   = S.VOL_MULT
    mode: str         = "breakout"   # or "pullback"
    hh_win: int       = S.HH_WIN     # breakout lookback (20/40/60)
    sma_win: int      = S.SMA_LONG   # trend filter (100/200)
    univ_size: int    = S.UNIV_SIZE  # top-ADV band size (50/100/150)
    mom_confirm: bool = False         # breakout: require positive lookback return
    rank_by: str      = "vol"         # "vol" (surge) or "mom" (lookback return)

    def label(self) -> str:
        m = "BO" if self.mode == "breakout" else "PB"
        mc = "+m" if self.mom_confirm else ""
        return (f"{m}{mc} hh{self.hh_win} sm{self.sma_win} u{self.univ_size} "
                f"N{self.slots} t{int(self.target_pct*100)} s{int(self.stop_pct*100)} "
                f"tr{int(self.trail_pct*100)} h{self.max_hold} v{self.vol_mult} r{self.rank_by}")


class Panels:
    """Numpy panels + precomputed config-independent masks (built once)."""
    def __init__(self, start: date, end: date):
        eng = _get_engine()
        n500 = [f"NSE:{s}-EQ" for s in sorted(universe_union("n500"))]
        print(f"N500 union pool (PIT): {len(n500)}")
        with eng.connect() as c:
            df = pd.read_sql(text(
                "SELECT symbol,date,open,high,low,close,volume FROM historical_data "
                "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b AND data_source='fyers' "
                "ORDER BY symbol,date"
            ), c, params={"s": n500, "a": start - timedelta(days=400), "b": end})
        df["date"] = pd.to_datetime(df["date"])
        df["adv_rs"] = df["close"].astype(float) * df["volume"].astype(float)

        STALE = 5
        all_dates = sorted(df["date"].unique())
        cutoff = all_dates[-(STALE + 1)] if len(all_dates) > STALE else all_dates[0]
        last_seen = df.groupby("symbol")["date"].max()
        fresh = set(last_seen[last_seen >= cutoff].index)

        cl  = df.pivot(index="date", columns="symbol", values="close").ffill()
        hi  = df.pivot(index="date", columns="symbol", values="high")
        vol = df.pivot(index="date", columns="symbol", values="volume")
        adv_rs = df.pivot(index="date", columns="symbol", values="adv_rs").fillna(0)

        # Multi-window precompute so a STRUCTURAL sweep can vary HH/SMA without
        # re-pulling. EMA + lookback momentum are single-window.
        HH_WINS  = [20, 40, 60]
        SMA_WINS = [100, 200]
        va20 = vol.rolling(20).mean()
        adv20 = adv_rs.rolling(S.ADV_WIN).mean()
        ema  = cl.ewm(span=PB_EMA, adjust=False).mean()
        ret_lb = cl / cl.shift(PB_LOOKBACK) - 1.0

        self.syms = list(cl.columns)
        self.col = {s: j for j, s in enumerate(self.syms)}
        self.dates = cl.index
        CL = cl.to_numpy(float)
        VOL = vol.to_numpy(float); VA = va20.to_numpy(float)
        EMA = ema.to_numpy(float); RET = ret_lb.to_numpy(float)
        self.CL = CL
        self.HHd  = {w: hi.rolling(w).max().shift(1).to_numpy(float) for w in HH_WINS}
        self.SMAd = {w: cl.rolling(w).mean().to_numpy(float) for w in SMA_WINS}
        with np.errstate(divide="ignore", invalid="ignore"):
            self.VR = np.where(VA > 0, VOL / VA, 0.0)
            self.ema_dist = np.where(EMA > 0, (CL - EMA) / EMA, np.nan)
        self.price_ok = (CL >= S.MIN_PRICE) & (CL <= S.MAX_PRICE)
        self.RET = RET
        self.va_ok = np.isfinite(VA) & (VA > 0)

        # yearly-PIT top-ADV band, FULL sorted column order (slice to univ_size).
        ys_list, cur = [], start
        while cur <= end:
            ys_list.append(pd.Timestamp(cur))
            try: cur = cur.replace(year=cur.year + 1)
            except ValueError: cur = cur.replace(year=cur.year + 1, day=28)
        self.year_starts = ys_list
        self.band_full: dict = {}
        for ys in ys_list:
            fut = self.dates[self.dates >= ys]
            if len(fut) == 0: continue
            di_ys = self.dates.get_loc(fut[0])
            elig = {f"NSE:{s}-EQ" for s in eligible_at("n500", fut[0].date())}
            a = adv20.iloc[di_ys].dropna().sort_values(ascending=False)
            a = a[a.index.isin(elig) & a.index.isin(fresh)]
            self.band_full[ys] = np.array([self.col[s] for s in a.index], int)

        self.trading_idx = [self.dates.get_loc(d) for d in self.dates
                            if start <= d.date() <= end]

    def band(self, di, univ_size):
        d = self.dates[di]
        chosen = self.year_starts[0]
        for ys in self.year_starts:
            if d >= ys: chosen = ys
        return self.band_full.get(chosen, np.array([], int))[:univ_size]

    def base_mask(self, cfg):
        """Build the entry-qualification mask (T×N bool) for one structural cfg."""
        HH = self.HHd[cfg.hh_win]; SMA = self.SMAd[cfg.sma_win]
        if cfg.mode == "breakout":
            m = (self.va_ok & np.isfinite(HH) & np.isfinite(SMA)
                 & (self.CL > HH) & (self.CL > SMA) & self.price_ok)
            if cfg.mom_confirm:
                m = m & np.isfinite(self.RET) & (self.RET > 0)   # already trending
        else:  # pullback
            m = (np.isfinite(SMA) & np.isfinite(self.ema_dist) & (self.CL > SMA)
                 & (self.ema_dist >= PB_LO) & (self.ema_dist <= PB_HI)
                 & np.isfinite(self.RET) & (self.RET * 100 >= PB_MOM_FLOOR) & self.price_ok)
        return m


def simulate(P: Panels, cfg: BacktestConfig, capital: float = DEFAULT_CAP):
    CL, VR = P.CL, P.VR
    base = P.base_mask(cfg)
    cap = capital
    held: dict = {}     # col_j -> [qty, entry_px, entry_di, peak]
    trades = []
    eqc = []

    for di in P.trading_idx:
        # 1) EXITS
        for j in list(held.keys()):
            close = CL[di, j]
            if not np.isfinite(close):
                continue
            pos = held[j]
            if close > pos[3]: pos[3] = close
            age = di - pos[2]
            reason = breakout_exit_reason(
                pos[1], float(close), pos[3], age,
                target_pct=cfg.target_pct, stop_pct=cfg.stop_pct,
                trail_pct=cfg.trail_pct, profit_trigger=cfg.profit_trig,
                max_hold_days=cfg.max_hold)
            if reason:
                exit_px = close * (1 - SLIP)
                proc = pos[0] * exit_px
                cap += proc - (proc * STT + BR)
                trades.append({
                    "sym": P.syms[j].replace("NSE:", "").replace("-EQ", ""),
                    "entry_date": P.dates[pos[2]].date().isoformat(),
                    "exit_date": P.dates[di].date().isoformat(),
                    "ret_pct": round((close - pos[1]) / pos[1] * 100, 2),
                    "pnl": round(proc - (proc * STT + BR) - pos[0] * pos[1], 0),
                    "reason": reason, "age_td": int(age),
                })
                del held[j]

        # 2) ENTRIES — vectorized daily scan over the PIT band
        free = cfg.slots - len(held)
        if free > 0:
            band = P.band(di, cfg.univ_size)
            if band.size:
                m = base[di, band]
                if cfg.mode == "breakout":
                    m = m & (VR[di, band] >= cfg.vol_mult)
                cols = band[m]
                cols = [c for c in cols if c not in held]
                if cols:
                    if cfg.rank_by == "mom":
                        cols.sort(key=lambda c: -P.RET[di, c])      # strongest momentum
                    elif cfg.mode == "breakout":
                        cols.sort(key=lambda c: -VR[di, c])        # strongest surge
                    else:
                        cols.sort(key=lambda c: P.ema_dist[di, c]) # closest to EMA (deepest dip)
                    picks = cols[:free]
                    alloc = cap / len(picks)
                    for c in picks:
                        entry_px = float(CL[di, c]) * (1 + SLIP)
                        q = int(alloc / entry_px)
                        if q >= 1 and q * entry_px + BR <= cap:
                            cap -= q * entry_px + BR
                            held[c] = [q, entry_px, di, entry_px]

        eq = cap + sum(pos[0] * CL[di, j] for j, pos in held.items()
                       if np.isfinite(CL[di, j]))
        eqc.append(eq)

    final = cap + sum(pos[0] * CL[-1, j] for j, pos in held.items()
                      if np.isfinite(CL[-1, j]))
    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = sum(1 for t in trades if t["pnl"] < 0)
    yrs = (P.dates[P.trading_idx[-1]].date() - P.dates[P.trading_idx[0]].date()).days / 365.25
    cagr = ((final / capital) ** (1 / yrs) - 1) * 100 if final > 0 else -100.0
    peak, mdd = capital, 0.0
    for eq in eqc:
        peak = max(peak, eq)
        mdd = max(mdd, (peak - eq) / peak * 100)
    by_year: dict = {}
    for di, eq in zip(P.trading_idx, eqc):
        by_year.setdefault(P.dates[di].year, []).append(eq)
    per_year, prev = {}, capital
    for y in sorted(by_year):
        per_year[str(y)] = round((by_year[y][-1] / prev - 1) * 100, 1)
        prev = by_year[y][-1]

    return {
        "final_nav": round(final, 0),
        "total_return_pct": round((final / capital - 1) * 100, 2),
        "cagr_pct": round(cagr, 2), "max_dd_pct": round(mdd, 2),
        "calmar": round(cagr / max(0.01, mdd), 2),
        "trades": len(trades), "wins": wins, "losses": losses,
        "win_rate_pct": round(wins / max(1, wins + losses) * 100, 1),
        "per_year": per_year, "years": round(yrs, 3), "_trades": trades,
    }


def sweep(P: Panels):
    grid = dict(slots=[1, 3, 5], target=[0.06, 0.10, 0.15], stop=[0.04, 0.06, 0.08],
                trail=[0.0, 0.05, 0.08], hold=[2, 3, 5], vol=[1.5, 2.0],
                mode=["breakout", "pullback"])
    results = []
    total = (len(grid["slots"]) * len(grid["target"]) * len(grid["stop"]) *
             len(grid["trail"]) * len(grid["hold"]) * len(grid["vol"]) * len(grid["mode"]))
    print(f"Sweeping {total} configs...")
    n = 0
    for mode in grid["mode"]:
      for sl in grid["slots"]:
        for tg in grid["target"]:
          for st in grid["stop"]:
            for tr in grid["trail"]:
              for hd in grid["hold"]:
                for vm in grid["vol"]:
                    cfg = BacktestConfig(slots=sl, target_pct=tg, stop_pct=st,
                                         trail_pct=tr, profit_trig=S.PROFIT_TRIG,
                                         max_hold=hd, vol_mult=vm, mode=mode)
                    r = simulate(P, cfg)
                    pos_yr = all(v > 0 for v in r["per_year"].values())
                    results.append({"label": cfg.label(), "cagr": r["cagr_pct"],
                                    "dd": r["max_dd_pct"], "calmar": r["calmar"],
                                    "tr": r["trades"], "wr": r["win_rate_pct"],
                                    "pos_yr": pos_yr, "per_year": r["per_year"]})
                    n += 1
                    if n % 50 == 0: print(f"  {n}/{total}")
    results.sort(key=lambda x: -x["calmar"])
    print("\n=== TOP 20 by Calmar ===")
    print(f"{'config':<30}{'CAGR%':>8}{'DD%':>7}{'Calmar':>8}{'Tr':>6}{'WR%':>6}{'+yr':>4}")
    for r in results[:20]:
        print(f"{r['label']:<30}{r['cagr']:>8.1f}{r['dd']:>7.1f}{r['calmar']:>8.2f}"
              f"{r['tr']:>6}{r['wr']:>6.1f}{'Y' if r['pos_yr'] else 'n':>4}")
    gate = [r for r in results if r["pos_yr"] and r["calmar"] >= 2.0]
    print(f"\n=== KILL-GATE survivors (positive every year AND Calmar>=2): {len(gate)} ===")
    for r in gate[:10]:
        print(f"{r['label']:<30} CAGR {r['cagr']:.1f}% DD {r['dd']:.1f}% Calmar {r['calmar']:.2f} | {r['per_year']}")
    if not gate:
        print("NONE clear the gate -> NO edge, do NOT ship (archive).")
    # also report best of each mode by CAGR for context
    for mode in ("BO", "PB"):
        mr = [r for r in results if r["label"].startswith(mode)]
        if mr:
            b = max(mr, key=lambda x: x["cagr"])
            print(f"\nbest {mode} by CAGR: {b['label']} -> {b['cagr']}% / DD {b['dd']}% / Calmar {b['calmar']} | {b['per_year']}")
    return results


def _report(results, tag):
    results.sort(key=lambda x: -x["calmar"])
    print(f"\n=== TOP 20 by Calmar [{tag}] ===")
    print(f"{'config':<46}{'CAGR%':>8}{'DD%':>7}{'Calmar':>8}{'Tr':>6}{'WR%':>6}{'+yr':>4}")
    for r in results[:20]:
        print(f"{r['label']:<46}{r['cagr']:>8.1f}{r['dd']:>7.1f}{r['calmar']:>8.2f}"
              f"{r['tr']:>6}{r['wr']:>6.1f}{'Y' if r['pos_yr'] else 'n':>4}")
    gate = [r for r in results if r["pos_yr"] and r["calmar"] >= 2.0]
    print(f"\n=== KILL-GATE survivors (positive every year AND Calmar>=2): {len(gate)} ===")
    for r in gate[:12]:
        print(f"{r['label']:<46} CAGR {r['cagr']:.1f}% DD {r['dd']:.1f}% Calmar {r['calmar']:.2f} | {r['per_year']}")
    if not gate:
        print("NONE clear the gate.")
    # best by CAGR regardless of gate (context)
    b = max(results, key=lambda x: x["cagr"])
    print(f"\nbest by CAGR: {b['label']} -> {b['cagr']}% / DD {b['dd']}% / Calmar {b['calmar']} | {b['per_year']}")


def sweep_struct(P: Panels):
    """STAGE A — vary STRUCTURE (hh/sma/univ/vol/mom_confirm/rank), fixed exit.

    Hunts for an entry-side edge before bothering to tune exits. Fixed exit =
    target15/stop8/trail5/hold5/slots3 (best-ish neighbourhood from the exit sweep).
    """
    EX = dict(target_pct=0.15, stop_pct=0.08, trail_pct=0.05,
              profit_trig=S.PROFIT_TRIG, max_hold=5, slots=3)
    results = []
    cfgs = []
    # breakout structural grid
    for hh in [20, 40, 60]:
        for sm in [100, 200]:
            for u in [50, 100, 150]:
                for vm in [2.0, 3.0, 4.0]:
                    for mc in [False, True]:
                        for rb in ["vol", "mom"]:
                            cfgs.append(BacktestConfig(mode="breakout", hh_win=hh, sma_win=sm,
                                        univ_size=u, vol_mult=vm, mom_confirm=mc, rank_by=rb, **EX))
    # pullback structural grid (hh/vol/mom_confirm N/A)
    for sm in [100, 200]:
        for u in [50, 100, 150]:
            for rb in ["vol", "mom"]:
                cfgs.append(BacktestConfig(mode="pullback", sma_win=sm, univ_size=u,
                                           rank_by=rb, **EX))
    print(f"Stage-A structural sweep: {len(cfgs)} configs...")
    for n, cfg in enumerate(cfgs, 1):
        r = simulate(P, cfg)
        results.append({"label": cfg.label(), "cagr": r["cagr_pct"], "dd": r["max_dd_pct"],
                        "calmar": r["calmar"], "tr": r["trades"], "wr": r["win_rate_pct"],
                        "pos_yr": all(v > 0 for v in r["per_year"].values()),
                        "per_year": r["per_year"]})
        if n % 40 == 0: print(f"  {n}/{len(cfgs)}")
    _report(results, "STRUCT")
    return results


def sweep_stageb(P: Panels):
    """STAGE B — lock the structural winner (breakout + mom rank, top-ADV leaders,
    strong surge), push TIGHTER (u30/40/50) + STRONGER (v4/5/6) and tune the exit
    grid. Hunts for a config that clears the kill-gate.
    """
    results = []
    cfgs = []
    for u in [30, 40, 50]:
        for vm in [4.0, 5.0, 6.0]:
            for sm in [100, 200]:
                for sl in [1, 3]:
                    for tg in [0.10, 0.15, 0.20]:
                        for st in [0.08, 0.10]:
                            for hd in [3, 5]:
                                cfgs.append(BacktestConfig(
                                    mode="breakout", rank_by="mom", hh_win=40, sma_win=sm,
                                    univ_size=u, vol_mult=vm, slots=sl, target_pct=tg,
                                    stop_pct=st, trail_pct=0.05, profit_trig=S.PROFIT_TRIG,
                                    max_hold=hd))
    print(f"Stage-B sweep: {len(cfgs)} configs...")
    for n, cfg in enumerate(cfgs, 1):
        r = simulate(P, cfg)
        results.append({"label": cfg.label(), "cagr": r["cagr_pct"], "dd": r["max_dd_pct"],
                        "calmar": r["calmar"], "tr": r["trades"], "wr": r["win_rate_pct"],
                        "pos_yr": all(v > 0 for v in r["per_year"].values()),
                        "per_year": r["per_year"]})
        if n % 60 == 0: print(f"  {n}/{len(cfgs)}")
    _report(results, "STAGE-B")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=DEFAULT_START.isoformat())
    ap.add_argument("--to", dest="end", default=DEFAULT_END.isoformat())
    ap.add_argument("--capital", type=float, default=DEFAULT_CAP)
    ap.add_argument("--stageb", action="store_true")
    ap.add_argument("--struct", action="store_true")
    ap.add_argument("--mode", default="breakout", choices=["breakout", "pullback"])
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    P = Panels(date.fromisoformat(a.start), date.fromisoformat(a.end))
    if a.stageb:
        sweep_stageb(P)
    elif a.struct:
        sweep_struct(P)
    elif a.sweep:
        sweep(P)
    else:
        r = simulate(P, BacktestConfig(mode=a.mode), a.capital)
        print(f"\n## breakout_swing_n500 ({BacktestConfig(mode=a.mode).label()})")
        print(f"  Final NAV:    Rs.{r['final_nav']:,.0f}")
        print(f"  Total return: {r['total_return_pct']:+.2f}%")
        print(f"  CAGR ({r['years']}y): {r['cagr_pct']:+.2f}%")
        print(f"  Trades: {r['trades']} (W={r['wins']}, L={r['losses']}, WR={r['win_rate_pct']}%)")
        print(f"  Max DD: {r['max_dd_pct']}%   Calmar: {r['calmar']}")
        print(f"  Per-year: {r['per_year']}")
