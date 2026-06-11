"""Price-level ENTRY/EXIT formula sweep on PIT Nifty-500 — 2025-03 -> now.

Goal (user): find a FORMULA that gives the right price to ENTER and the right
price to EXIT for N500 stocks, then prove it with a backtest 2025-03 -> now.

Three formula families, each level computed from PRIOR-bar (di-1) data only
(no lookahead); the trigger is an intraday touch judged against the day's
OHLC (daily-bar proxy), fills modelled conservatively:

  MR  (mean-reversion / buy-the-dip):
        entry_level = SMA_n - k*ATR        buy if day low touches it
        exit        = SMA_n  OR  entry*(1+tp)   sell on target
        stop        = entry - s*ATR
  BRK (breakout / buy strength):
        entry_level = N-day Donchian high   buy if day high breaks it
        exit        = M-day Donchian low  OR  k*ATR trailing stop
        stop        = entry - s*ATR
  ATR (close-signal bracket):
        entry = close (di) if close>SMA50 & 20d-mom>0
        target= entry + tt*ATR   stop = entry - ss*ATR

Capital model: TOP-N ranked, equal-weight. When more names trigger than free
slots, rank by a score (deepest dip / highest momentum) and take the best.

Run inside the app container:
  docker exec trading_system_app python tools/research/price_formula_lab.py \
      --from 2025-03-01 --to 2026-06-10 --topn 5
"""
import sys, argparse, itertools
from pathlib import Path
from datetime import date, timedelta

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from sqlalchemy import text
from tools.shared.ohlcv_cache import _get_engine
from tools.shared.index_membership import universe_union, eligible_at

COST = 0.0015           # per side (matches the live models' backtests)
DEFAULT_START = date(2025, 3, 1)
DEFAULT_END = date(2026, 6, 10)
CAP = 1_000_000.0


# ----------------------------------------------------------------------------
# Data load
# ----------------------------------------------------------------------------
def load_panels(eng, start, end):
    syms = [f"NSE:{s}-EQ" for s in sorted(universe_union("n500"))]
    with eng.connect() as c:
        df = pd.read_sql(text(
            "SELECT symbol,date,open,high,low,close,volume FROM historical_data "
            "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b AND data_source='fyers' "
            "ORDER BY symbol,date"
        ), c, params={"s": syms, "a": start - timedelta(days=420), "b": end})
    df["date"] = pd.to_datetime(df["date"])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    piv = lambda v: df.pivot(index="date", columns="symbol", values=v).sort_index()
    C = piv("close").ffill()
    O = piv("open"); H = piv("high"); L = piv("low"); V = piv("volume")
    # align all to C's grid
    O = O.reindex_like(C); H = H.reindex_like(C); L = L.reindex_like(C); V = V.reindex_like(C)
    return C, O, H, L, V


def build_eligibility(dates, cols):
    """date x symbol bool: was the name a PIT N500 member that day."""
    plain = [s.replace("NSE:", "").replace("-EQ", "") for s in cols]
    elig = np.zeros((len(dates), len(cols)), dtype=bool)
    cache = {}
    for i, d in enumerate(dates):
        dd = d.date()
        key = (dd.year, dd.month)           # membership snapshots are monthly-ish
        s = cache.get(key)
        if s is None:
            s = eligible_at("n500", dd); cache[key] = s
        elig[i] = [p in s for p in plain]
    return elig


def atr(H, L, C, n=14):
    pc = C.shift(1)
    tr = np.maximum.reduce([(H - L).values, (H - pc).abs().values, (L - pc).abs().values])
    tr = pd.DataFrame(tr, index=C.index, columns=C.columns)
    return tr.rolling(n, min_periods=n).mean()


def rsi(C, n=14):
    d = C.diff()
    up = d.clip(lower=0).rolling(n, min_periods=n).mean()
    dn = (-d).clip(lower=0).rolling(n, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# ----------------------------------------------------------------------------
# Engine — one config replay
# ----------------------------------------------------------------------------
def replay(cfg, P, i0, i1, topn, capital):
    """P = dict of numpy arrays. Returns nav Series + trades list."""
    C, O, H, L, ELIG = P["C"], P["O"], P["H"], P["L"], P["ELIG"]
    ENTRY, RANK = P[cfg["entry_key"]], P[cfg["rank_key"]]
    ATRv = P["ATR"]
    SMA = P.get(cfg.get("sma_key"))           # MR target
    DLOW = P.get(cfg.get("dlow_key"))         # BRK donchian-low exit
    fam = cfg["fam"]
    s_atr = cfg.get("stop_atr"); tp = cfg.get("tp"); trail = cfg.get("trail")
    tt = cfg.get("target_atr"); maxhold = cfg.get("maxhold")
    cool = cfg.get("cooldown", 0); tgt_off = cfg.get("tgt_off", 0.0)
    min_hold = cfg.get("min_hold", 0)
    adv_floor = cfg.get("adv_floor", 0.0); ADV = P.get("adv20")
    min_price = cfg.get("min_price", 0.0)
    exit_mode = cfg.get("exit")                 # sma | rsi_ob | bbhigh
    rsi_ob = cfg.get("rsi_ob"); RSI = P.get("rsi14")
    BBHI = P.get("bbhi"); rsi_in = cfg.get("rsi_entry_max")
    close_fill = cfg.get("close_fill", False)   # live-parity: all fills at CLOSE
    SMA200 = P.get("sma200")                    # optional uptrend entry gate
    dates = P["dates"]; ncol = C.shape[1]

    cash = capital; pos = {}; trades = []; navs = []; nd = []; last_exit = {}
    for di in range(i0, i1):
        o, h, l, c = O[di], H[di], L[di], C[di]
        atr_p = ATRv[di - 1]
        # ---------------- EXITS (check open positions against today OHLC) ----
        for s in list(pos.keys()):
            p = pos[s]; px = None; why = None
            cp = c[s]
            if np.isnan(cp):
                continue
            # update trailing peak
            if trail is not None:
                p["peak"] = max(p["peak"], h[s] if not np.isnan(h[s]) else p["peak"])
                tstop = p["peak"] - trail * (atr_p[s] if not np.isnan(atr_p[s]) else 0)
            stop = p["stop"]; tgt = p["target"]; held = di - p["in_di"]
            # STOP (assume stop fills before target on a both-hit day = conservative).
            # min_hold only gates the PROFIT exits (target/trail/time), never the stop.
            if not np.isnan(stop) and l[s] <= stop:
                px = min(o[s], stop) if not np.isnan(o[s]) else stop; why = "STOP"
            elif held < min_hold:
                px = None
            elif trail is not None and not np.isnan(tstop) and l[s] <= tstop:
                px = min(o[s], tstop) if not np.isnan(o[s]) else tstop; why = "TRAIL"
            elif fam == "BRK" and DLOW is not None and not np.isnan(DLOW[di-1][s]) and l[s] <= DLOW[di-1][s]:
                dl = DLOW[di-1][s]; px = min(o[s], dl) if not np.isnan(o[s]) else dl; why = "DCH_LOW"
            elif exit_mode == "rsi_ob" and RSI is not None and not np.isnan(RSI[di][s]) and RSI[di][s] >= rsi_ob:
                px = cp; why = "RSI_OB"            # overbought: sell at the close
            elif exit_mode == "bbhigh" and BBHI is not None and not np.isnan(BBHI[di-1][s]) and h[s] >= BBHI[di-1][s]:
                bh = BBHI[di-1][s]; px = max(o[s], bh) if not np.isnan(o[s]) else bh; why = "BB_HIGH"
            elif tgt is not None and not np.isnan(tgt) and h[s] >= tgt:
                px = max(o[s], tgt) if not np.isnan(o[s]) else tgt; why = "TARGET"
            elif maxhold is not None and held >= maxhold:
                px = cp; why = "TIME"
            if px is not None:
                if close_fill:
                    px = cp                      # market order at the close
                cash += p["qty"] * px * (1 - COST)
                trades.append((s, p["entry"], px, p["in_di"], di, why))
                last_exit[s] = di; del pos[s]
        # ---------------- ENTRIES --------------------------------------------
        free = topn - len(pos)
        if free > 0:
            lvl = ENTRY[di - 1]                       # level from PRIOR bar
            cand = []
            for s in range(ncol):
                if s in pos or not ELIG[di][s]:
                    continue
                if cool and s in last_exit and (di - last_exit[s]) < cool:
                    continue                              # re-entry cooldown
                if adv_floor > 0 and ADV is not None:
                    av = ADV[di - 1][s]
                    if np.isnan(av) or av < adv_floor:    # liquidity gate (20d ADV)
                        continue
                if min_price > 0:
                    pc = C[di - 1][s]                     # penny-whipsaw gate (ORB lesson)
                    if np.isnan(pc) or pc < min_price:
                        continue
                if rsi_in is not None and RSI is not None:
                    rv = RSI[di - 1][s]                    # oversold-confirmation gate
                    if np.isnan(rv) or rv > rsi_in:
                        continue
                if cfg.get("trend_gate") and SMA200 is not None:
                    sv = SMA200[di - 1][s]                 # uptrend gate: close>SMA200
                    if np.isnan(sv) or C[di - 1][s] <= sv:
                        continue
                Lv = lvl[s]
                if np.isnan(Lv) or np.isnan(atr_p[s]) or atr_p[s] <= 0:
                    continue
                fill = None
                if fam == "MR":                       # buy limit below: low must touch
                    if not np.isnan(l[s]) and l[s] <= Lv:
                        fill = min(o[s], Lv) if not np.isnan(o[s]) else Lv
                        if close_fill:
                            fill = c[s]               # market order at the close
                elif fam == "BRK":                    # buy stop above: high must break
                    if not np.isnan(h[s]) and h[s] >= Lv:
                        fill = max(o[s], Lv) if not np.isnan(o[s]) else Lv
                elif fam == "ATR":                    # close-signal: enter at close
                    if Lv > 0 and not np.isnan(c[s]):
                        fill = c[s]
                if fill is None or np.isnan(fill) or fill <= 0:
                    continue
                rk = RANK[di - 1][s]
                if np.isnan(rk):
                    continue
                if cfg.get("mom_floor") is not None and rk < cfg["mom_floor"]:
                    continue                          # require minimum momentum
                cand.append((rk, s, fill, atr_p[s]))
            cand.sort(reverse=True)                   # highest score first
            for rk, s, fill, a in cand:
                if free <= 0:
                    break
                q = int((cash / max(1, free)) / fill)
                if q < 1:
                    continue
                cash -= q * fill * (1 + COST)
                stop = fill - s_atr * a if s_atr is not None else np.nan
                if fam == "MR":
                    if cfg.get("exit") == "sma" and SMA is not None:
                        tgt = SMA[di-1][s] + tgt_off * a   # ride past mean by tgt_off*ATR
                    elif tp is not None:
                        tgt = fill * (1 + tp)
                    else:
                        tgt = np.nan                       # trail-only exit
                elif fam == "ATR":
                    tgt = fill + tt * a
                else:
                    tgt = np.nan
                pos[s] = {"qty": q, "entry": fill, "in_di": di,
                          "stop": stop, "target": tgt, "peak": fill}
                free -= 1
        # ---------------- mark NAV -------------------------------------------
        mv = cash + sum(p["qty"] * C[di][s] for s, p in pos.items() if not np.isnan(C[di][s]))
        navs.append(mv); nd.append(dates[di])
    return pd.Series(navs, index=pd.DatetimeIndex(nd)), trades


def score(nav, capital, start, end):
    final = float(nav.iloc[-1])
    yrs = (end - start).days / 365.25
    cagr = ((final / capital) ** (1 / yrs) - 1) * 100 if final > 0 else -100.0
    roll = nav.cummax(); mdd = float(((roll - nav) / roll).max()) * 100
    return cagr, mdd, cagr / max(0.5, mdd), final


# ----------------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=DEFAULT_START.isoformat())
    ap.add_argument("--to", dest="end", default=DEFAULT_END.isoformat())
    ap.add_argument("--topn", type=int, default=5)
    ap.add_argument("--fam", default="all")
    a = ap.parse_args()
    start = date.fromisoformat(a.start); end = date.fromisoformat(a.end)
    eng = _get_engine()
    C, O, H, L, V = load_panels(eng, start, end)
    dates = C.index
    print(f"panel: {C.shape[0]} days x {C.shape[1]} syms  {dates[0].date()}..{dates[-1].date()}")
    ELIG = build_eligibility(dates, C.columns)

    # --- indicators (pandas, vectorized) -> float32 numpy, free frames asap ---
    import gc
    npf = lambda x: x.values.astype("float32")
    ATRv = atr(H, L, C, 14)
    sma20 = C.rolling(20, min_periods=20).mean()
    sma50 = C.rolling(50, min_periods=50).mean()
    P = {"C": npf(C), "O": npf(O), "H": npf(H), "L": npf(L), "ATR": npf(ATRv),
         "dates": dates, "ELIG": ELIG,
         "sma20": npf(sma20), "sma50": npf(sma50),
         "mom60": npf(C / C.shift(60) - 1), "mom120": npf(C / C.shift(120) - 1),
         "mom30": npf(C / C.shift(30) - 1),
         "dip20": npf((sma20 - C) / ATRv), "dip50": npf((sma50 - C) / ATRv),
         "adv20": npf((C * V).rolling(20, min_periods=20).mean())}   # 20d avg traded value
    if a.fam in ("all", "MR", "CFV", "OPT", "OPT2", "PF"):
        for sk, sma in (("20", sma20), ("50", sma50)):
            for k in (1.0, 1.5, 2.0, 2.5):
                P[f"mrlvl{sk}_{k}"] = npf(sma - k * ATRv)
    if a.fam == "OPT2":
        for k in (0.5, 0.75):
            P[f"mrlvl50_{k}"] = npf(sma50 - k * ATRv)
        P["sma200"] = npf(C.rolling(200, min_periods=200).mean())
        vol60 = C.pct_change().rolling(60, min_periods=60).std()
        P["volad60"] = npf((C / C.shift(60) - 1) / vol60)
        del vol60
    if a.fam in ("all", "BRK"):
        P["dch20"] = npf(H.rolling(20, min_periods=20).max())
        P["dch55"] = npf(H.rolling(55, min_periods=55).max())
        P["dlow10"] = npf(L.rolling(10, min_periods=10).min())
        P["dlow20"] = npf(L.rolling(20, min_periods=20).min())
    if a.fam in ("all", "ATR"):
        mom20 = C / C.shift(20) - 1
        P["atrsig"] = npf(((C > sma50) & (mom20 > 0)).astype(float))
        del mom20
    if a.fam in ("all", "ENH"):
        std20 = C.rolling(20, min_periods=20).std()
        P["rsi14"] = npf(rsi(C, 14))
        P["bbhi"] = npf(sma20 + 2 * std20)        # upper Bollinger (overbought band)
        P["bblo"] = npf(sma20 - 2 * std20)        # lower Bollinger = support entry level
        for k in (1.0, 1.5):                        # MR ATR-dip entry levels for ENH too
            P[f"mrlvl50_{k}"] = npf(sma50 - k * ATRv)
        del std20
    del C, O, H, L, V, ATRv, sma20, sma50
    gc.collect()

    i0 = dates.searchsorted(pd.Timestamp(start))
    i1 = dates.searchsorted(pd.Timestamp(end), side="right")
    warm = 130
    i0 = max(i0, warm)

    cfgs = []
    # ---- MR family: CHURN-REDUCTION sweep around the winning skeleton ----
    # Fixed winners: sma50 entry, exit-at-mean, mom60 rank, 1.5xATR stop, mh40.
    # Vary the levers that cut trade count: deeper entry (k), ride-past-mean
    # target (tgt_off), re-entry cooldown, min-hold, ADV liquidity floor.
    if a.fam in ("all", "MR"):
        for k in (1.0, 1.5, 2.0):
            for off in (0.0, 1.0, 2.0):
                for cd in (0, 5, 10):
                    for mhld in (0, 5):
                        for advf in (0.0, 2.5e8, 1.0e9):
                            cfgs.append(dict(
                                fam="MR", entry_key=f"mrlvl50_{k}",
                                sma_key="sma50", rank_key="mom60",
                                exit="sma", tgt_off=off, stop_atr=1.5, maxhold=40,
                                cooldown=cd, min_hold=mhld, adv_floor=advf,
                                tag=f"MR k{k} off{off} cd{cd} mh{mhld} "
                                    f"adv{int(advf/1e7)}cr"))
    # ---- BRK family ----
    if a.fam in ("all", "BRK"):
        for dk in ("dch20", "dch55"):
            for exk in ("dlow10", "dlow20", "trail3"):
                for s_atr in (2.0, 3.0):
                    for rankk in ("mom60", "mom120"):
                        cfgs.append(dict(
                            fam="BRK", entry_key=dk, rank_key=rankk,
                            dlow_key=(exk if exk.startswith("dlow") else None),
                            trail=(3.0 if exk == "trail3" else None),
                            stop_atr=s_atr, maxhold=None,
                            tag=f"BRK {dk} exit{exk} stop{s_atr} rk{rankk}"))
    # ---- OPT2: the UNSWEPT dimensions (user recheck, 2026-06-11) ------------
    # shallow dips (k<1), SMA200 uptrend gate, vol-adj rank, no-stop,
    # cross-target (sma20 exit on sma50 entry anchor), tp8 at topn3.
    if a.fam == "OPT2":
        for k in (0.5, 0.75, 1.0):
            for gate in (False, True):
                for rkk in ("mom60", "volad60"):
                    for stp in (1.5, None):
                        for tgt in ("sma50", "sma20", "tp8"):
                            cfgs.append(dict(
                                fam="MR", entry_key=f"mrlvl50_{k}",
                                sma_key=("sma20" if tgt == "sma20" else "sma50"),
                                rank_key=rkk,
                                exit=(None if tgt == "tp8" else "sma"),
                                tp=(0.08 if tgt == "tp8" else None),
                                tgt_off=0.0, stop_atr=stp, maxhold=40,
                                cooldown=10, trend_gate=gate,
                                tag=f"OPT2 k{k} g{int(gate)} {rkk} "
                                    f"st{stp or 'NO'} x{tgt}"))
    # ---- OPT: CAGR-improvement grid around the BUILT model (topn3 winner) ----
    # Fixed: k1.0 entry, exit@SMA, limit fills. Vary: rank window, momentum
    # floor, target offset, stop, maxhold, cooldown. Gate downstream: a change
    # only counts if it improves BOTH the recent AND the full-cycle window.
    if a.fam == "OPT":
        for rk in ("mom30", "mom60", "mom120"):
            for floor in (None, 0.20):
                for off in (0.0, 1.0):
                    for stp in (1.5, 2.0):
                        for mh in (20, 40):
                            for cd in (10, 15):
                                cfgs.append(dict(
                                    fam="MR", entry_key="mrlvl50_1.0",
                                    sma_key="sma50", rank_key=rk,
                                    exit="sma", tgt_off=off, stop_atr=stp,
                                    maxhold=mh, cooldown=cd, mom_floor=floor,
                                    tag=f"OPT {rk} fl{floor or 0} off{off} "
                                        f"st{stp} mh{mh} cd{cd}"))
    # ---- PF: price filter + momentum-speed sweep (user ask, 2026-06-11) ----
    # Unswept dims: min entry price (penny-whipsaw gate) x intermediate momentum
    # floors (0 / .05 / .10 raw-return; .20 was too strict = 9 trades) x faster
    # rank window (mom30 vs mom60). Skeleton fixed = built model (k1.0, exit@SMA,
    # st1.5, mh40, cd10). Deploy gate: must improve BOTH windows.
    if a.fam == "PF":
        for mp in (0.0, 50.0, 100.0, 200.0):
            for floor in (None, 0.0, 0.05, 0.10):
                for rk in ("mom30", "mom60"):
                    cfgs.append(dict(
                        fam="MR", entry_key="mrlvl50_1.0",
                        sma_key="sma50", rank_key=rk,
                        exit="sma", tgt_off=0.0, stop_atr=1.5,
                        maxhold=40, cooldown=10,
                        mom_floor=floor, min_price=mp,
                        tag=f"PF mp{int(mp)} fl{'N' if floor is None else floor} {rk}"))
    # ---- CFV: close-fill (live-parity) verification of the winner config ----
    if a.fam == "CFV":
        for cd in (0, 5, 10):
            for cf in (False, True):
                cfgs.append(dict(
                    fam="MR", entry_key="mrlvl50_1.0", sma_key="sma50",
                    rank_key="mom60", exit="sma", tgt_off=0.0,
                    stop_atr=1.5, maxhold=40, cooldown=cd, close_fill=cf,
                    tag=f"CFV k1.0 cd{cd} fill={'close' if cf else 'limit'}"))
        cfgs.append(dict(                          # close-fill deeper-dip variant
            fam="MR", entry_key="mrlvl50_1.5", sma_key="sma50",
            rank_key="mom60", exit="sma", tgt_off=0.0,
            stop_atr=1.5, maxhold=40, cooldown=10, close_fill=True,
            tag="CFV k1.5 cd10 fill=close"))
    # ---- ENH family: better entries (ATR-dip / lower-Bollinger, +RSI gate) x
    #      better exits (mean / overbought RSI / upper-Bollinger). All PIT (di-1). ----
    if a.fam in ("all", "ENH"):
        entries = [("atr1.0", "mrlvl50_1.0"), ("atr1.5", "mrlvl50_1.5"),
                   ("bblo", "bblo")]
        for ename, ekey in entries:
            for rin in (None, 35):                 # optional oversold-confirm gate
                # exit options: ride-to-mean(+off), RSI overbought, upper band
                exits = [("sma0", dict(exit="sma", tgt_off=0.0)),
                         ("sma2", dict(exit="sma", tgt_off=2.0)),
                         ("rsi70", dict(exit="rsi_ob", rsi_ob=70)),
                         ("rsi75", dict(exit="rsi_ob", rsi_ob=75)),
                         ("bbhi", dict(exit="bbhigh"))]
                for xname, xkw in exits:
                    cfgs.append(dict(
                        fam="MR", entry_key=ekey, sma_key="sma50",
                        rank_key="mom60", stop_atr=1.5, maxhold=40,
                        cooldown=5, min_hold=3,
                        rsi_entry_max=rin, **xkw,
                        tag=f"ENH {ename}{'+rsi35' if rin else ''} x{xname}"))
    # ---- ATR family ----  entry-signal panel "atrsig" built above (close>sma50 & mom20>0)
    if a.fam in ("all", "ATR"):
        for tt in (2.0, 3.0, 4.0):
            for ss in (1.5, 2.0):
                for mh in (20, 40):
                    for rankk in ("mom60", "mom120"):
                        cfgs.append(dict(
                            fam="ATR", entry_key="atrsig", rank_key=rankk,
                            target_atr=tt, stop_atr=ss, maxhold=mh,
                            tag=f"ATR tgt{tt} stop{ss} mh{mh} rk{rankk}"))

    print(f"sweeping {len(cfgs)} configs, topn={a.topn} ...")
    results = []
    for cfg in cfgs:
        try:
            nav, trades = replay(cfg, P, i0, i1, a.topn, CAP)
            if len(nav) < 5:
                continue
            cagr, mdd, cal, final = score(nav, CAP, start, end)
            wins = sum(1 for t in trades if t[2] > t[1])
            results.append((cal, cagr, mdd, len(trades),
                            100*wins/max(1, len(trades)), cfg["tag"], nav, trades))
        except Exception as e:
            print(f"  FAIL {cfg['tag']}: {e}")
    results.sort(key=lambda r: -r[0])
    with open("/tmp/lab_results.csv", "w") as f:
        f.write("calmar,cagr,dd,trades,wr,tag\n")
        for cal, cagr, mdd, ntr, wr, tag, _, _ in results:
            f.write(f"{cal:.2f},{cagr:.1f},{mdd:.1f},{ntr},{wr:.0f},{tag}\n")
    print("\n=== TOP 15 by Calmar ===")
    print(f"{'Calmar':>6} {'CAGR%':>7} {'DD%':>6} {'trades':>6} {'WR%':>5}  config")
    for cal, cagr, mdd, ntr, wr, tag, nav, trades in results[:15]:
        print(f"{cal:6.2f} {cagr:7.1f} {mdd:6.1f} {ntr:6d} {wr:5.0f}  {tag}")
    print("\n=== TOP 8 by CAGR ===")
    for cal, cagr, mdd, ntr, wr, tag, nav, trades in sorted(results, key=lambda r:-r[1])[:8]:
        print(f"{cal:6.2f} {cagr:7.1f} {mdd:6.1f} {ntr:6d} {wr:5.0f}  {tag}")
    span_yrs = max(0.1, (end - start).days / 365.25)
    for cap in (150, 90, 50):
        sub = [r for r in results if r[3] <= cap]
        if not sub:
            continue
        print(f"\n=== LOW-CHURN: best by Calmar with <= {cap} trades "
              f"(<= {cap/span_yrs:.0f}/yr) ===")
        for cal, cagr, mdd, ntr, wr, tag, nav, trades in sub[:5]:
            print(f"{cal:6.2f} {cagr:7.1f} {mdd:6.1f} {ntr:6d} {wr:5.0f}  {tag}")

    if results:
        best = results[0]
        nav = best[6]
        print(f"\n=== BEST by Calmar: {best[5]} ===")
        print(f"  CAGR {best[1]:+.1f}%  MaxDD {best[2]:.1f}%  Calmar {best[0]:.2f}  "
              f"trades {best[3]} (WR {best[4]:.0f}%)  finalNAV Rs.{float(nav.iloc[-1]):,.0f}")
        for yy, g in nav.groupby(nav.index.year):
            if len(g) < 2: continue
            rl = g.cummax(); dd = float(((rl-g)/rl).max())*100
            print(f"    {yy}: {(g.iloc[-1]/g.iloc[0]-1)*100:+.1f}%  (DD {dd:.1f}%)")


if __name__ == "__main__":
    main()
