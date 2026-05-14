"""
5%-target profit-booking strategy backtest across Nifty 100.
Strategies:
  1) 52w_high_breakout
  2) ema20_pullback
  3) gap_up_continuation
  4) bull_engulfing_support
  5) higher_high_breakout
"""
import json, math, sys, time
from collections import defaultdict
from datetime import date, datetime
import psycopg
import numpy as np
import pandas as pd

DB = "postgresql://trader:trader_password@database:5432/trading_system"
START = date(2024, 5, 13)
END   = date(2026, 5, 12)
CAPITAL = 1_000_000.0
MAX_POS = 5
COST = 0.0013  # 0.13% round-trip
TARGET = 0.05
UNIVERSE_FILE = "/app/logs/momrot/universes/n100_current.json"

def fyers_sym(s):
    return f"NSE:{s}-EQ"

def load_universe():
    with open(UNIVERSE_FILE) as f:
        d = json.load(f)
    return [fyers_sym(x["symbol"]) for x in d["stocks"]]

def load_bars(syms):
    conn = psycopg.connect(DB)
    # need ~310 trading days of warmup before START for 250-day high
    warmup_start = date(2023, 1, 1)
    q = """SELECT symbol, date, open, high, low, close, volume
           FROM historical_data
           WHERE symbol = ANY(%s) AND date >= %s AND date <= %s
           ORDER BY symbol, date"""
    with conn.cursor() as cur:
        cur.execute(q, (syms, warmup_start, END))
        rows = cur.fetchall()
    conn.close()
    df = pd.DataFrame(rows, columns=["symbol","date","open","high","low","close","volume"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol","date"]).reset_index(drop=True)
    return df

def add_indicators(g):
    g = g.copy()
    g["ema20"] = g["close"].ewm(span=20, adjust=False).mean()
    g["ema50"] = g["close"].ewm(span=50, adjust=False).mean()
    g["vol20"] = g["volume"].rolling(20).mean()
    g["hh20"]  = g["high"].rolling(20).max().shift(1)  # prior 20-day high
    g["hh250"] = g["high"].rolling(250).max().shift(1) # prior 250-day high
    return g

# ---- signal functions: produce list of (entry_date, entry_price, stop_pct, target_pct, max_hold_days)
def sig_52w_breakout(g):
    sigs = []
    for i in range(len(g)):
        if i < 251: continue
        row = g.iloc[i]
        if pd.isna(row["hh250"]) or pd.isna(row["vol20"]): continue
        if row["close"] > row["hh250"] and row["volume"] > 1.5 * row["vol20"]:
            sigs.append((row["date"], row["close"], 0.03, 0.05, 15))
    return sigs

def sig_ema20_pullback(g):
    sigs = []
    for i in range(len(g)):
        if i < 50: continue
        row = g.iloc[i]
        if any(pd.isna([row["ema20"], row["ema50"]])): continue
        if row["close"] > row["ema50"] and row["ema20"] > row["ema50"] \
           and row["low"] <= row["ema20"] and row["close"] > row["ema20"]:
            sigs.append((row["date"], row["close"], 0.02, 0.05, 10))
    return sigs

def sig_gap_up(g):
    sigs = []
    for i in range(1, len(g)):
        if i < 50: continue
        row = g.iloc[i]; prev = g.iloc[i-1]
        if pd.isna(prev["ema20"]): continue
        if row["open"] > prev["close"] * 1.02 and prev["close"] > prev["ema20"]:
            sigs.append((row["date"], row["open"], 0.02, 0.05, 5))
    return sigs

def sig_bull_engulfing(g):
    sigs = []
    for i in range(1, len(g)):
        if i < 50: continue
        row = g.iloc[i]; prev = g.iloc[i-1]
        if pd.isna(row["ema50"]): continue
        bullish_engulfing = (prev["close"] < prev["open"]) and (row["close"] > row["open"]) \
                            and (row["open"] <= prev["close"]) and (row["close"] >= prev["open"])
        if not bullish_engulfing: continue
        dist = (row["close"] - row["ema50"]) / row["ema50"]
        if 0 <= dist <= 0.05:
            sigs.append((row["date"], row["close"], 0.02, 0.05, 7))
    return sigs

def sig_hh_breakout(g):
    sigs = []
    for i in range(1, len(g)):
        if i < 20: continue
        row = g.iloc[i]
        if pd.isna(row["hh20"]) or pd.isna(row["vol20"]): continue
        if row["high"] > row["hh20"] and row["close"] > row["open"] and row["volume"] > 1.3 * row["vol20"]:
            sigs.append((row["date"], row["close"], 0.02, 0.05, 7))
    return sigs

STRATS = {
    "52w_high_breakout":   sig_52w_breakout,
    "ema20_pullback":      sig_ema20_pullback,
    "gap_up_continuation": sig_gap_up,
    "bull_engulfing_support": sig_bull_engulfing,
    "higher_high_breakout": sig_hh_breakout,
}

def simulate_trade(g, entry_idx, entry_price, stop_pct, target_pct, max_hold, use_open_entry=False):
    """Walk forward from entry. Return (exit_date, exit_price, holding_days, outcome)
    outcome: 'target', 'stop', 'time'."""
    stop_price = entry_price * (1 - stop_pct)
    target_price = entry_price * (1 + target_pct)
    # entry day: if use_open_entry, check intraday this same day; else next bar
    start = entry_idx if use_open_entry else entry_idx + 1
    end = min(len(g), entry_idx + max_hold + 1)
    for j in range(start, end):
        bar = g.iloc[j]
        # conservative: if both hit same day, assume stop first
        if bar["low"] <= stop_price:
            return (bar["date"], stop_price, j - entry_idx, "stop")
        if bar["high"] >= target_price:
            return (bar["date"], target_price, j - entry_idx, "target")
    # time exit at close of last bar
    last = g.iloc[end - 1]
    return (last["date"], last["close"], end - 1 - entry_idx, "time")

def run_strategy(name, sig_fn, df_all):
    """Portfolio-level sim: cap=10L, 5 concurrent slots, equal-weight per entry."""
    # Build per-symbol signals first
    sigs_per_sym = {}
    for sym, g in df_all.groupby("symbol", sort=False):
        g = add_indicators(g.reset_index(drop=True))
        sigs = sig_fn(g)
        # filter to backtest window for entry
        sigs = [(d,p,s,t,h,i) for i,(d,p,s,t,h) in
                ( (g.index[g["date"]==d][0] if (g["date"]==d).any() else None, (d,p,s,t,h))
                  for (d,p,s,t,h) in sigs )
                if i is not None and START <= d.date() <= END]
        sigs_per_sym[sym] = (g, sigs)

    # Flat list sorted by entry date
    all_sigs = []
    for sym, (g, sigs) in sigs_per_sym.items():
        for (d,p,s,t,h,idx) in sigs:
            all_sigs.append((d, sym, idx, p, s, t, h))
    all_sigs.sort(key=lambda x: x[0])

    open_positions = {}  # sym -> (exit_date(Timestamp), exit_price, qty, entry_price)
    cash = CAPITAL
    trades = []
    use_open = (name == "gap_up_continuation")
    def _to_ts(x):
        return x if isinstance(x, pd.Timestamp) else pd.Timestamp(x)

    # Process in chronological order; close positions whose exit_date <= current entry_date BEFORE opening
    for sig in all_sigs:
        entry_date, sym, idx, p_entry, sl, tgt, mh = sig
        entry_date_ts = _to_ts(entry_date)
        # close any pos that should have exited by entry_date
        for s in list(open_positions.keys()):
            ed, ep, qty, entry_p = open_positions[s]
            ed_ts = _to_ts(ed)
            if ed_ts <= entry_date_ts:
                pnl = qty * (ep - entry_p) - (qty * entry_p + qty * ep) * COST / 2
                cash += qty * ep - qty * ep * COST / 2
                trades.append({
                    "sym": s, "entry": entry_p, "exit": ep, "qty": qty,
                    "pnl": pnl, "ret_pct": (ep/entry_p - 1),
                    "exit_date": ed_ts,
                })
                del open_positions[s]
        if sym in open_positions: continue
        if len(open_positions) >= MAX_POS: continue
        slot = cash / (MAX_POS - len(open_positions))
        if slot < 1000: continue
        qty = int(slot // p_entry)
        if qty <= 0: continue
        g, _ = sigs_per_sym[sym]
        ed, ep, hd, outcome = simulate_trade(g, idx, p_entry, sl, tgt, mh, use_open_entry=use_open)
        cost_buy = qty * p_entry * COST / 2
        cash -= qty * p_entry + cost_buy
        open_positions[sym] = (_to_ts(ed), ep, qty, p_entry)

    # Close everything remaining
    for s, (ed, ep, qty, entry_p) in open_positions.items():
        pnl = qty * (ep - entry_p) - (qty * entry_p + qty * ep) * COST / 2
        cash += qty * ep - qty * ep * COST / 2
        trades.append({"sym": s, "entry": entry_p, "exit": ep, "qty": qty,
                       "pnl": pnl, "ret_pct": (ep/entry_p - 1), "exit_date": _to_ts(ed)})
    return trades, cash

def metrics(trades, final_cash, name):
    if not trades:
        return {"strategy": name, "trades": 0, "win_rate": 0, "avg_pnl": 0,
                "total_roi_pct": (final_cash/CAPITAL-1)*100, "monthly_roi_pct": 0,
                "maxdd_pct": 0, "sharpe": 0, "target_hit_rate": 0}
    target_hits = sum(1 for t in trades if t["ret_pct"] >= 0.0495)  # ~5%
    wins = sum(1 for t in trades if t["pnl"] > 0)
    avg_pnl = np.mean([t["pnl"] for t in trades])
    total_roi = (final_cash / CAPITAL - 1) * 100
    months = 24
    monthly_roi = total_roi / months
    # daily equity curve approx: bucket by exit_date
    by_date = defaultdict(float)
    for t in trades:
        by_date[t["exit_date"]] += t["pnl"]
    days = sorted(by_date.keys())
    eq = CAPITAL
    daily_rets = []
    peak = CAPITAL
    maxdd = 0
    for d in days:
        eq += by_date[d]
        daily_rets.append(by_date[d] / CAPITAL)
        peak = max(peak, eq)
        dd = (peak - eq) / peak
        maxdd = max(maxdd, dd)
    sharpe = 0
    if len(daily_rets) > 1 and np.std(daily_rets) > 0:
        sharpe = np.mean(daily_rets) / np.std(daily_rets) * math.sqrt(252)
    return {
        "strategy": name,
        "trades": len(trades),
        "win_rate": wins / len(trades) * 100,
        "target_hit_rate": target_hits / len(trades) * 100,
        "avg_pnl": avg_pnl,
        "total_roi_pct": total_roi,
        "monthly_roi_pct": monthly_roi,
        "maxdd_pct": maxdd * 100,
        "sharpe": sharpe,
    }

def main():
    syms = load_universe()
    print(f"[i] universe size: {len(syms)}", flush=True)
    print(f"[i] loading bars...", flush=True)
    t0 = time.time()
    df = load_bars(syms)
    print(f"[i] loaded {len(df)} rows in {time.time()-t0:.1f}s", flush=True)
    print(f"[i] unique syms in db: {df['symbol'].nunique()}", flush=True)

    results = []
    per_strat_trades = {}
    for name, fn in STRATS.items():
        t0 = time.time()
        print(f"[*] running {name}...", flush=True)
        trades, cash = run_strategy(name, fn, df)
        m = metrics(trades, cash, name)
        results.append(m)
        per_strat_trades[name] = trades
        print(f"    -> {m['trades']} trades, win {m['win_rate']:.1f}%, target {m['target_hit_rate']:.1f}%, ROI {m['total_roi_pct']:.1f}%, DD {m['maxdd_pct']:.1f}%  ({time.time()-t0:.1f}s)", flush=True)

    print("\n=== COMPARISON TABLE ===")
    print(f"{'strategy':<25} {'trades':>6} {'win%':>6} {'tgt%':>6} {'ROI%':>7} {'mo%':>6} {'DD%':>6} {'Sharpe':>7}")
    for m in results:
        print(f"{m['strategy']:<25} {m['trades']:>6d} {m['win_rate']:>6.1f} {m['target_hit_rate']:>6.1f} {m['total_roi_pct']:>7.1f} {m['monthly_roi_pct']:>6.2f} {m['maxdd_pct']:>6.1f} {m['sharpe']:>7.2f}")

    # Winner = strategy w/ best target_hit_rate AND positive ROI
    eligible = [m for m in results if m["total_roi_pct"] > 0 and m["trades"] >= 20]
    if eligible:
        winner = max(eligible, key=lambda x: x["target_hit_rate"])
        print(f"\n[winner] {winner['strategy']} (target hit {winner['target_hit_rate']:.1f}%, ROI {winner['total_roi_pct']:.1f}%)")
        # Top-10 stocks for winner
        wt = per_strat_trades[winner["strategy"]]
        per_sym = defaultdict(lambda: {"trades":0, "tgts":0, "pnl":0.0, "ret_sum":0.0})
        for t in wt:
            ps = per_sym[t["sym"]]
            ps["trades"] += 1
            if t["ret_pct"] >= 0.0495: ps["tgts"] += 1
            ps["pnl"] += t["pnl"]
            ps["ret_sum"] += t["ret_pct"]
        top = sorted(per_sym.items(), key=lambda kv: (-(kv[1]["tgts"]/max(kv[1]["trades"],1)), -kv[1]["trades"]))
        print(f"\n[top 10 stocks for {winner['strategy']}]")
        shown = 0
        for sym, st in top:
            if st["trades"] < 3: continue
            print(f"  {sym:<22} trades={st['trades']:>3} target_hits={st['tgts']:>3} ({st['tgts']/st['trades']*100:>5.1f}%) total_pnl=Rs{st['pnl']:>10,.0f}")
            shown += 1
            if shown >= 10: break
    else:
        print("\n[!] NO strategy with positive ROI + sufficient trades")

    # dump details
    out = {"results": results, "winner": eligible and max(eligible, key=lambda x: x["target_hit_rate"])["strategy"]}
    with open("/tmp/backtest_5pct_results.json","w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\n[done] /tmp/backtest_5pct_results.json")

if __name__ == "__main__":
    main()
