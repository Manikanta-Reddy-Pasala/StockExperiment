"""5-MINUTE execution validation for price_meanrev_n500 — 2025-03 -> now.

The model's backtest (102.8% CAGR / 12.2% DD) judges intraday touches against
DAILY bars (low<=level => filled at level; stop-before-target on both-hit
days). This script replays the same PIT levels against REAL 5-minute paths to
measure what the daily proxy can't see:

  V1  backtest-semantics on 5-min path: any eligible toucher can fill, but
      TIME priority (first trade-through wins a slot, not EOD momentum pick),
      fills require trading THROUGH the limit (low < level - 1 tick; an exact
      touch = no fill), exits on the 5-min poll grid (bar close = polled LTP,
      natural slippage past the level), same-day exits allowed.
  V2  live-exact semantics: only the top-(K-held) momentum names get resting
      orders each morning (what fyers_executor_limit --place actually does);
      everything else as V1. THE go/no-go number for funding the model.
  V0  daily-proxy reproduction (sanity: must print ~102.8 / 12.2).

Diagnostics: touch-only (no-trade-through) rate, both-hit-day true ordering
(target-first vs stop-first), exit slippage vs level, V2 placed-order fill
rate, paper-ledger (15 backups) vs executor (3 orders) candidate gap.

Phases (run inside the app container; 5m bars cached as pickle under
/app/logs/research_5min/ so re-runs don't re-pull):
  python tools/research/fivemin_validation.py --phase plan     # symbol list
  python tools/research/fivemin_validation.py --phase pull     # fetch 5m bars
  python tools/research/fivemin_validation.py --phase replay   # V0+V1+V2
"""
import sys, json, time, argparse, logging
from pathlib import Path
from datetime import date, datetime

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

# silence the api_calls logger — it dumps every 5m candle as JSON otherwise
logging.getLogger("api_calls").setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("fivemin_val")

from tools.research.price_formula_lab import load_panels, build_eligibility, atr
from tools.shared.ohlcv_cache import _get_engine

START = date(2025, 3, 1)
END = date(2026, 6, 10)
CAP = 1_000_000.0
COST = 0.0015
K = 3; SMA_LEN = 50; ATR_LEN = 14; ENTRY_K = 1.0
STOP_ATR = 1.5; MAXHOLD = 40; COOLDOWN = 10
TICK = 0.05
CACHE = Path("/app/logs/research_5min")
PLAN_FILE = CACHE / "plan.json"
MIN_OF_DAY0 = 9 * 60 + 15            # first 5m bar 09:15
N_SLOTS = 75                          # 09:15..15:25 inclusive


def daily_context():
    """Load the daily panel + PIT levels exactly like the lab/backtest."""
    eng = _get_engine()
    C, O, H, L, V = load_panels(eng, START, END)
    dates = C.index
    ELIG = build_eligibility(dates, C.columns)
    ATRv = atr(H, L, C, ATR_LEN)
    sma50 = C.rolling(SMA_LEN, min_periods=SMA_LEN).mean()
    lvl = sma50 - ENTRY_K * ATRv
    mom60 = C / C.shift(60) - 1
    f32 = lambda x: x.values.astype("float32")
    P = dict(C=f32(C), O=f32(O), H=f32(H), L=f32(L),
             ATR=f32(ATRv), SMA=f32(sma50), LVL=f32(lvl), MOM=f32(mom60),
             ELIG=ELIG, dates=dates,
             cols=list(C.columns))
    i0 = max(int(dates.searchsorted(pd.Timestamp(START))), 130)
    i1 = int(dates.searchsorted(pd.Timestamp(END), side="right"))
    return P, i0, i1


# ---------------------------------------------------------------------------
# V0 — daily-proxy replay (the backtest's own semantics; sanity check)
# ---------------------------------------------------------------------------
def replay_daily(P, i0, i1):
    C, O, H, L = P["C"], P["O"], P["H"], P["L"]
    LVL, MOM, SMA, ATRv, ELIG = P["LVL"], P["MOM"], P["SMA"], P["ATR"], P["ELIG"]
    dates = P["dates"]; ncol = C.shape[1]
    cash = CAP; pos = {}; trades = []; navs = []; nd = []; last_exit = {}
    for di in range(i0, i1):
        o, h, l, c = O[di], H[di], L[di], C[di]
        for s in list(pos.keys()):
            p = pos[s]; px = None; why = None
            cp = c[s]
            if np.isnan(cp):
                continue
            if l[s] <= p["stop"]:
                px = min(o[s], p["stop"]) if not np.isnan(o[s]) else p["stop"]; why = "STOP"
            elif h[s] >= p["target"]:
                px = max(o[s], p["target"]) if not np.isnan(o[s]) else p["target"]; why = "TARGET"
            elif di - p["in_di"] >= MAXHOLD:
                px = cp; why = "TIME"
            if px is not None:
                cash += p["qty"] * px * (1 - COST)
                trades.append((s, p["entry"], px, p["in_di"], di, why))
                last_exit[s] = di; del pos[s]
        free = K - len(pos)
        if free > 0:
            lvl = LVL[di - 1]; atr_p = ATRv[di - 1]
            cand = []
            for s in range(ncol):
                if s in pos or not ELIG[di][s]:
                    continue
                if s in last_exit and (di - last_exit[s]) < COOLDOWN:
                    continue
                Lv = lvl[s]
                if np.isnan(Lv) or Lv <= 0 or np.isnan(atr_p[s]) or atr_p[s] <= 0:
                    continue
                if np.isnan(l[s]) or l[s] > Lv:
                    continue
                rk = MOM[di - 1][s]
                if np.isnan(rk):
                    continue
                fill = min(o[s], Lv) if not np.isnan(o[s]) else Lv
                cand.append((rk, s, fill, atr_p[s]))
            cand.sort(reverse=True)
            for rk, s, fill, a in cand:
                if free <= 0:
                    break
                q = int((cash / max(1, free)) / fill)
                if q < 1:
                    continue
                cash -= q * fill * (1 + COST)
                pos[s] = {"qty": q, "entry": fill, "in_di": di,
                          "stop": fill - STOP_ATR * a, "target": SMA[di - 1][s]}
                free -= 1
        mv = cash + sum(p["qty"] * C[di][s] for s, p in pos.items() if not np.isnan(C[di][s]))
        navs.append(mv); nd.append(dates[di])
    return pd.Series(navs, index=pd.DatetimeIndex(nd)), trades


# ---------------------------------------------------------------------------
# plan — which symbols ever DAILY-touch their level in the window
# ---------------------------------------------------------------------------
def phase_plan():
    P, i0, i1 = daily_context()
    L, LVL, ELIG = P["L"], P["LVL"], P["ELIG"]
    need = set()
    for di in range(i0, i1):
        lv = LVL[di - 1]; lo = L[di]
        m = ELIG[di] & ~np.isnan(lv) & (lv > 0) & ~np.isnan(lo) & (lo <= lv)
        for s in np.nonzero(m)[0]:
            need.add(P["cols"][s])
    CACHE.mkdir(parents=True, exist_ok=True)
    PLAN_FILE.write_text(json.dumps(sorted(need)))
    log.info(f"plan: {len(need)} symbols ever daily-touch their level "
             f"({(i1-i0)} days) -> {PLAN_FILE}")


# ---------------------------------------------------------------------------
# pull — fetch + cache 5m bars for the planned symbols
# ---------------------------------------------------------------------------
def phase_pull():
    from src.services.brokers.fyers_service import FyersService
    syms = json.loads(PLAN_FILE.read_text())
    svc = FyersService()
    done = 0; failed = []
    for sym in syms:
        out = CACHE / f"{sym.replace(':', '_')}.pkl"
        if out.exists():
            done += 1; continue
        rows = []
        cursor = START
        try:
            while cursor <= END:
                chunk_end = min(date.fromordinal(cursor.toordinal() + 94), END)
                r = svc.history(1, sym, "NSE", "5m",
                                cursor.isoformat(), chunk_end.isoformat())
                for cd in (r or {}).get("data", {}).get("candles", []) or []:
                    rows.append((int(cd["timestamp"]), float(cd["open"]),
                                 float(cd["high"]), float(cd["low"]),
                                 float(cd["close"])))
                cursor = date.fromordinal(chunk_end.toordinal() + 1)
                time.sleep(0.25)
            df = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c"]).drop_duplicates("ts")
            df.to_pickle(out)
            done += 1
            if done % 25 == 0:
                log.info(f"pull: {done}/{len(syms)}")
        except Exception as e:
            failed.append(sym); log.warning(f"pull FAIL {sym}: {e}")
            time.sleep(2)
    log.info(f"pull complete: {done}/{len(syms)} ok, {len(failed)} failed: {failed[:10]}")


# ---------------------------------------------------------------------------
# 5-min replay (V1 backtest-semantics / V2 live-exact)
# ---------------------------------------------------------------------------
def load_5m_dense(sym, day_to_row, n_rows):
    """-> float32 [n_rows, N_SLOTS, 4] (o,h,l,c; NaN = no bar) or None.

    Dense per-symbol cube keyed by replay-day row (di - i0). The raw pickle
    DataFrames (python-string day per row) OOM the 1GiB container at 573
    symbols; this is ~375KB/symbol fixed.
    """
    f = CACHE / f"{sym.replace(':', '_')}.pkl"
    if not f.exists():
        return None
    df = pd.read_pickle(f)
    ts = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
    day = ts.dt.strftime("%Y-%m-%d").map(day_to_row)
    slot = ((ts.dt.hour * 60 + ts.dt.minute) - MIN_OF_DAY0) // 5
    ok = day.notna() & (slot >= 0) & (slot < N_SLOTS)
    arr = np.full((n_rows, N_SLOTS, 4), np.nan, dtype="float32")
    arr[day[ok].astype(int).values, slot[ok].values] = \
        df.loc[ok, ["o", "h", "l", "c"]].values.astype("float32")
    return arr


def replay_5min(P, i0, i1, cache5, live_exact):
    """live_exact=False -> V1 (any toucher, time priority);
       live_exact=True  -> V2 (orders only for top-(K-held) momentum names)."""
    C, O, L, Hd = P["C"], P["O"], P["L"], P["H"]
    LVL, MOM, SMA, ATRv, ELIG = P["LVL"], P["MOM"], P["SMA"], P["ATR"], P["ELIG"]
    dates = P["dates"]; cols = P["cols"]; ncol = C.shape[1]
    cash = CAP; pos = {}; trades = []; navs = []; nd = []; last_exit = {}
    diag = dict(touch_only=0, fills=0, placed=0, both_hit_target_first=0,
                both_hit_stop_first=0, stop_slip=[], same_day_exits=0)
    for di in range(i0, i1):
        row = di - i0
        lvl = LVL[di - 1]; atr_p = ATRv[di - 1]; sma_p = SMA[di - 1]; mom_p = MOM[di - 1]
        # ---- today's order book ------------------------------------------
        cands = []
        for s in range(ncol):
            if s in pos or not ELIG[di][s]:
                continue
            if s in last_exit and (di - last_exit[s]) < COOLDOWN:
                continue
            Lv = lvl[s]
            if np.isnan(Lv) or Lv <= 0 or np.isnan(atr_p[s]) or atr_p[s] <= 0:
                continue
            rk = mom_p[s]
            if np.isnan(rk):
                continue
            cands.append((rk, s, float(Lv)))
        cands.sort(reverse=True)
        free0 = K - len(pos)
        if live_exact:
            orders = {s: Lv for rk, s, Lv in cands[:max(0, free0)]}
        else:
            orders = {s: Lv for rk, s, Lv in cands}
        diag["placed"] += min(len(orders), free0) if live_exact else 0
        mom_of = {s: rk for rk, s, Lv in cands}
        # 5m day-rows for relevant symbols (held + orders that daily-touched)
        def _day(s):
            cube = cache5.get(cols[s])
            if cube is None:
                return None
            dr = cube[row]
            return dr if not np.all(np.isnan(dr[:, 3])) else None
        bars = {}
        for s in list(pos.keys()):
            bars[s] = _day(s)
        for s, Lv in orders.items():
            if not np.isnan(L[di][s]) and L[di][s] <= Lv:   # daily touch happened
                bars[s] = _day(s)
        filled_today = set(); touched_today = set()
        # MAXHOLD fallback when a held name has no 5m data today (failed pull):
        # live would still time-exit via the daily-counted holding age.
        for s in list(pos.keys()):
            p = pos[s]
            if bars.get(s) is None and di - p["in_di"] >= MAXHOLD:
                cp = C[di][s]
                if not np.isnan(cp):
                    cash += p["qty"] * cp * (1 - COST)
                    trades.append((s, p["entry"], cp, p["in_di"], di, "TIME"))
                    last_exit[s] = di; del pos[s]
        # ---- walk the 75 slots -------------------------------------------
        for slot in range(N_SLOTS):
            # exits first (poll = bar close)
            for s in list(pos.keys()):
                p = pos[s]
                b = bars.get(s)
                if b is None or np.isnan(b[slot][3]):
                    continue
                bc = float(b[slot][3])
                px = None; why = None
                if bc <= p["stop"]:
                    px = bc; why = "STOP"
                    diag["stop_slip"].append((p["stop"] - bc) / p["stop"])
                elif bc >= p["target"]:
                    px = bc; why = "TARGET"
                elif di - p["in_di"] >= MAXHOLD:
                    px = bc; why = "TIME"
                if px is not None:
                    if p["in_di"] == di:
                        diag["same_day_exits"] += 1
                    # both-hit day truth: daily bar pierced BOTH stop and target
                    if (not np.isnan(L[di][s]) and L[di][s] <= p["stop"]
                            and not np.isnan(Hd[di][s]) and Hd[di][s] >= p["target"]):
                        diag["both_hit_target_first" if why == "TARGET"
                             else "both_hit_stop_first"] += 1
                    cash += p["qty"] * px * (1 - COST)
                    trades.append((s, p["entry"], px, p["in_di"], di, why))
                    last_exit[s] = di; del pos[s]
            # entries (resting limits live from 09:16 -> slot >= 1)
            if slot >= 1 and len(pos) < K:
                hits = []
                for s, Lv in orders.items():
                    if s in pos or s in filled_today:
                        continue
                    b = bars.get(s)
                    if b is None or np.isnan(b[slot][3]):
                        continue
                    bo, bh, bl, bc = (float(x) for x in b[slot])
                    fill = None
                    if bo < Lv:                       # gapped/trading below: fill at mkt
                        fill = bo
                    elif bl <= Lv - TICK:             # traded THROUGH the limit
                        fill = Lv
                    elif bl <= Lv:                    # exact touch — no fill; order rests on
                        if s not in touched_today:    # (a later trade-through can still fill)
                            diag["touch_only"] += 1
                            touched_today.add(s)
                        continue
                    if fill is None or fill <= 0:
                        continue
                    hits.append((mom_of.get(s, -9e9), s, fill))
                hits.sort(reverse=True)               # same-bar tie -> momentum
                for rk, s, fill in hits:
                    free = K - len(pos)
                    if free <= 0:
                        break
                    q = int((cash / max(1, free)) / fill)
                    if q < 1:
                        continue
                    cash -= q * fill * (1 + COST)
                    pos[s] = {"qty": q, "entry": fill, "in_di": di,
                              "stop": fill - STOP_ATR * atr_p[s],
                              "target": sma_p[s]}
                    filled_today.add(s); diag["fills"] += 1
        mv = cash + sum(p["qty"] * C[di][s] for s, p in pos.items() if not np.isnan(C[di][s]))
        navs.append(mv); nd.append(dates[di])
    return pd.Series(navs, index=pd.DatetimeIndex(nd)), trades, diag


def stats(label, nav, trades):
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = ((nav.iloc[-1] / CAP) ** (1 / max(yrs, .1)) - 1) * 100
    roll = nav.cummax(); mdd = float(((roll - nav) / roll).max()) * 100
    wins = sum(1 for t in trades if t[2] > t[1])
    why = {}
    for t in trades:
        why[t[5]] = why.get(t[5], 0) + 1
    log.info(f"{label}: CAGR {cagr:+.1f}%  DD {mdd:.1f}%  Calmar {cagr/max(.5,mdd):.2f}  "
             f"trades {len(trades)} (WR {100*wins/max(1,len(trades)):.0f}%)  exits {why}")
    for yy, g in nav.groupby(nav.index.year):
        if len(g) < 2:
            continue
        rl = g.cummax(); dd = float(((rl - g) / rl).max()) * 100
        log.info(f"    {yy}: {(g.iloc[-1]/g.iloc[0]-1)*100:+.1f}%  (DD {dd:.1f}%)")
    return cagr, mdd


def phase_replay():
    P, i0, i1 = daily_context()
    nav0, tr0 = replay_daily(P, i0, i1)
    stats("V0 daily-proxy (sanity, expect ~102.8/12.2)", nav0, tr0)

    import gc
    syms = json.loads(PLAN_FILE.read_text())
    day_to_row = {d.date().isoformat(): di - i0
                  for di, d in enumerate(P["dates"]) if di >= i0}
    n_rows = i1 - i0
    cache5 = {}
    miss = 0
    for j, sym in enumerate(syms):
        cube = load_5m_dense(sym, day_to_row, n_rows)
        if cube is None:
            miss += 1
        cache5[sym] = cube
        if (j + 1) % 100 == 0:
            gc.collect()
    log.info(f"5m cache: {len(syms)-miss}/{len(syms)} symbols loaded ({miss} missing)")

    nav1, tr1, dg1 = replay_5min(P, i0, i1, cache5, live_exact=False)
    stats("V1 5-min path, backtest candidates (time-priority fills)", nav1, tr1)
    log.info(f"  diag V1: fills {dg1['fills']}  touch-only(no fill) {dg1['touch_only']}  "
             f"same-day exits {dg1['same_day_exits']}  "
             f"both-hit days: target-first {dg1['both_hit_target_first']} vs "
             f"stop-first {dg1['both_hit_stop_first']}  "
             f"stop slip avg {100*np.mean(dg1['stop_slip'] or [0]):.2f}% (n={len(dg1['stop_slip'])})")

    nav2, tr2, dg2 = replay_5min(P, i0, i1, cache5, live_exact=True)
    stats("V2 LIVE-EXACT (top-(K-held) resting orders only)", nav2, tr2)
    log.info(f"  diag V2: orders placed {dg2['placed']}  fills {dg2['fills']} "
             f"(fill rate {100*dg2['fills']/max(1,dg2['placed']):.0f}%)  "
             f"touch-only {dg2['touch_only']}  same-day exits {dg2['same_day_exits']}  "
             f"stop slip avg {100*np.mean(dg2['stop_slip'] or [0]):.2f}%")
    out = CACHE / "replay_results.json"
    out.write_text(json.dumps({
        "v0_nav_final": float(nav0.iloc[-1]), "v1_nav_final": float(nav1.iloc[-1]),
        "v2_nav_final": float(nav2.iloc[-1]),
        "v1_diag": {k: (v if not isinstance(v, list) else len(v)) for k, v in dg1.items()},
        "v2_diag": {k: (v if not isinstance(v, list) else len(v)) for k, v in dg2.items()},
    }, indent=2))
    log.info(f"results -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["plan", "pull", "replay"])
    a = ap.parse_args()
    {"plan": phase_plan, "pull": phase_pull, "replay": phase_replay}[a.phase]()
