"""
Positional backtests: 4 models on Nifty 100 over 2023-05-13 to 2026-05-12.
Capital 10L each, 0.13% round-trip cost, equal-weight per slot.
Uses spot OHLCV only.
"""
import psycopg, json
import pandas as pd
import numpy as np
from datetime import date, timedelta
from collections import defaultdict
import math

# -------------------- Config --------------------
START = date(2023, 5, 13)
END = date(2026, 5, 12)
CAPITAL = 1_000_000.0   # 10L per model
COST_RT = 0.0013        # 0.13% round-trip
MAX_SLOTS = 5

# -------------------- Load data --------------------
print("Loading data...", flush=True)
conn = psycopg.connect("postgresql://trader:trader_password@database:5432/trading_system")
with open("/app/logs/momrot/universes/n100_current.json") as f:
    uni = json.load(f)
SYMBOLS = [f"NSE:{s['symbol']}-EQ" for s in uni["stocks"]]

# Need 18 months lookback before START for 250-day high / 12w return etc.
LOAD_START = START - timedelta(days=600)
cur = conn.cursor()
cur.execute(
    """select symbol, date, open, high, low, close, volume
       from historical_data
       where symbol = ANY(%s) and date >= %s and date <= %s
       order by symbol, date""",
    (SYMBOLS, LOAD_START, END),
)
rows = cur.fetchall()
df = pd.DataFrame(rows, columns=["symbol", "date", "open", "high", "low", "close", "volume"])
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
print(f"Loaded {len(df):,} bars / {df['symbol'].nunique()} symbols, "
      f"{df['date'].min().date()} → {df['date'].max().date()}", flush=True)

# Pivots
close_pv = df.pivot(index="date", columns="symbol", values="close").sort_index()
open_pv = df.pivot(index="date", columns="symbol", values="open").sort_index()
high_pv = df.pivot(index="date", columns="symbol", values="high").sort_index()
low_pv = df.pivot(index="date", columns="symbol", values="low").sort_index()
vol_pv = df.pivot(index="date", columns="symbol", values="volume").sort_index()

ALL_DATES = list(close_pv.index)
TRADE_DATES = [d for d in ALL_DATES if d.date() >= START and d.date() <= END]
print(f"Trading dates: {len(TRADE_DATES)}", flush=True)


# -------------------- Helpers --------------------
def annualize(total_return, n_days):
    yrs = n_days / 365.25
    if yrs <= 0:
        return 0.0
    return (1 + total_return) ** (1 / yrs) - 1


def max_drawdown(equity):
    peak = equity.cummax()
    dd = equity / peak - 1
    return dd.min()


def sharpe(daily_returns):
    if daily_returns.std() == 0 or len(daily_returns) < 2:
        return 0.0
    return (daily_returns.mean() / daily_returns.std()) * math.sqrt(252)


def yearly_returns(equity_series):
    """Return dict of year -> pct return."""
    res = {}
    df_eq = equity_series.copy()
    df_eq.index = pd.to_datetime(df_eq.index)
    for yr in [2023, 2024, 2025, 2026]:
        sub = df_eq[df_eq.index.year == yr]
        if len(sub) < 2:
            continue
        ret = sub.iloc[-1] / sub.iloc[0] - 1
        res[yr] = ret * 100
    return res


def run_portfolio(trades, all_dates):
    """
    trades: list of dicts {entry_date, exit_date, symbol, entry_px, exit_px, slot}
    Build daily equity curve assuming MAX_SLOTS independent slots, each starting at CAPITAL/MAX_SLOTS.
    Capital compounds within slot. Cost applied round-trip on exit.
    """
    slot_cap = CAPITAL / MAX_SLOTS

    # Build per-slot timeline
    slot_equity = {s: [slot_cap] for s in range(MAX_SLOTS)}
    slot_dates = {s: [all_dates[0]] for s in range(MAX_SLOTS)}
    slot_balance = [slot_cap] * MAX_SLOTS  # current cash if flat
    # slot active position: (symbol, qty, entry_cash)
    slot_pos = [None] * MAX_SLOTS

    trades_by_entry = defaultdict(list)
    trades_by_exit = defaultdict(list)
    for t in trades:
        trades_by_entry[pd.Timestamp(t["entry_date"])].append(t)
        trades_by_exit[pd.Timestamp(t["exit_date"])].append(t)

    daily_equity_total = []
    daily_dates = []

    for d in all_dates:
        # process exits first
        for t in trades_by_exit.get(d, []):
            s = t["slot"]
            if slot_pos[s] is not None and slot_pos[s][0] == t["symbol"]:
                qty = slot_pos[s][1]
                proceeds = qty * t["exit_px"]
                # apply round-trip cost on exit proceeds (entry cost already deducted at entry)
                proceeds *= (1 - COST_RT / 2)
                slot_balance[s] = proceeds
                slot_pos[s] = None

        # process entries
        for t in trades_by_entry.get(d, []):
            s = t["slot"]
            if slot_pos[s] is None:
                # buy at entry_px with current slot_balance
                cash = slot_balance[s] * (1 - COST_RT / 2)
                qty = cash / t["entry_px"]
                slot_pos[s] = (t["symbol"], qty, slot_balance[s])
                slot_balance[s] = 0.0

        # MTM
        total = 0.0
        for s in range(MAX_SLOTS):
            if slot_pos[s] is not None:
                sym, qty, _ = slot_pos[s]
                px = close_pv.at[d, sym] if sym in close_pv.columns else np.nan
                if pd.isna(px):
                    # last known
                    px = close_pv[sym].loc[:d].dropna().iloc[-1] if sym in close_pv.columns else 0
                total += qty * px
            else:
                total += slot_balance[s]
        daily_equity_total.append(total)
        daily_dates.append(d)

    eq = pd.Series(daily_equity_total, index=daily_dates)
    return eq


def summarize(name, trades, equity, m3_yearly):
    if len(trades) == 0:
        print(f"\n=== {name} ===\nNO TRADES")
        return None
    n_days = (equity.index[-1] - equity.index[0]).days
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = annualize(total_ret, n_days)
    daily_ret = equity.pct_change().dropna()
    sh = sharpe(daily_ret)
    mdd = max_drawdown(equity)
    holds = [(pd.Timestamp(t["exit_date"]) - pd.Timestamp(t["entry_date"])).days for t in trades]
    avg_hold = np.mean(holds)
    pnl_pct = [(t["exit_px"] / t["entry_px"] - 1) * 100 - COST_RT * 100 for t in trades]
    wins = [p for p in pnl_pct if p > 0]
    losses = [p for p in pnl_pct if p <= 0]
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    win_rate = len(wins) / len(pnl_pct) * 100
    yr = yearly_returns(equity)

    # Correlation with M3 yearly ROIs
    m3_vals = []
    self_vals = []
    for y, v in m3_yearly.items():
        if y in yr:
            m3_vals.append(v)
            self_vals.append(yr[y])
    corr = np.corrcoef(m3_vals, self_vals)[0, 1] if len(m3_vals) >= 2 else float("nan")

    out = dict(name=name, cagr=cagr * 100, mdd=mdd * 100, sharpe=sh,
               n_trades=len(trades), avg_hold=avg_hold,
               avg_win=avg_win, avg_loss=avg_loss, win_rate=win_rate,
               yearly=yr, m3_corr=corr)

    print(f"\n=== {name} ===")
    print(f"  CAGR: {cagr*100:.2f}%  MaxDD: {mdd*100:.2f}%  Sharpe: {sh:.2f}")
    print(f"  Trades: {len(trades)}  AvgHold: {avg_hold:.0f} days  WinRate: {win_rate:.1f}%")
    print(f"  AvgWin: {avg_win:.2f}%  AvgLoss: {avg_loss:.2f}%")
    print(f"  Yearly: {yr}")
    print(f"  M3 corr: {corr:.3f}")
    return out


# -------------------- Common utilities --------------------
def find_first_trade_day_of_quarter(year):
    qs = [date(year, 1, 1), date(year, 4, 1), date(year, 7, 1), date(year, 10, 1)]
    out = []
    for q in qs:
        # find earliest TRADE_DATE >= q
        for d in TRADE_DATES:
            if d.date() >= q:
                out.append(d)
                break
    return out


QUARTER_DAYS = sorted(set(
    find_first_trade_day_of_quarter(2023) + find_first_trade_day_of_quarter(2024) +
    find_first_trade_day_of_quarter(2025) + find_first_trade_day_of_quarter(2026)
))
QUARTER_DAYS = [d for d in QUARTER_DAYS if d.date() >= START and d.date() <= END]
print(f"Quarter rebalance days: {[d.date() for d in QUARTER_DAYS]}", flush=True)


# ===================================================================
# Model 1: Weekly momentum quarterly rotation (12-week return)
# ===================================================================
def model_weekly_momentum():
    trades = []
    for i, rebal_d in enumerate(QUARTER_DAYS):
        # 12 weeks ~ 60 trading days lookback
        lookback_idx = ALL_DATES.index(rebal_d) - 60
        if lookback_idx < 0:
            continue
        lookback_d = ALL_DATES[lookback_idx]
        # Compute 12w return for each symbol
        rets = {}
        for sym in close_pv.columns:
            c_now = close_pv.at[rebal_d, sym]
            c_then = close_pv.at[lookback_d, sym]
            if pd.notna(c_now) and pd.notna(c_then) and c_then > 0:
                rets[sym] = c_now / c_then - 1
        ranked = sorted(rets.items(), key=lambda x: -x[1])[:MAX_SLOTS]
        top_syms = [s for s, _ in ranked]

        # exit date = next quarter day (or END)
        if i + 1 < len(QUARTER_DAYS):
            exit_d = QUARTER_DAYS[i + 1]
        else:
            exit_d = TRADE_DATES[-1]

        for slot, sym in enumerate(top_syms):
            # entry at next day open
            entry_idx = ALL_DATES.index(rebal_d) + 1
            if entry_idx >= len(ALL_DATES):
                continue
            entry_d = ALL_DATES[entry_idx]
            entry_px = open_pv.at[entry_d, sym]
            if pd.isna(entry_px):
                continue
            # exit at exit_d open
            exit_idx_d = exit_d if exit_d in ALL_DATES else TRADE_DATES[-1]
            exit_px = open_pv.at[exit_idx_d, sym]
            if pd.isna(exit_px):
                exit_px = close_pv[sym].loc[:exit_idx_d].dropna().iloc[-1]
            trades.append(dict(entry_date=entry_d, exit_date=exit_idx_d, symbol=sym,
                               entry_px=entry_px, exit_px=exit_px, slot=slot))
    return trades


# ===================================================================
# Model 2: 52w breakout position (250-day high + volume + trail SL)
# ===================================================================
def model_52w_breakout():
    trades = []
    active_slots = [None] * MAX_SLOTS  # slot -> (sym, entry_d, entry_px, peak, vol_avg)
    vol_sma20 = vol_pv.rolling(20).mean()
    hh250 = high_pv.rolling(250).max().shift(1)  # prior 250d high

    for i in range(len(TRADE_DATES)):
        d = TRADE_DATES[i]
        # update trailing SL / exits for active
        for slot in range(MAX_SLOTS):
            if active_slots[slot] is None:
                continue
            sym, e_date, e_px, peak, days_held = active_slots[slot]
            close_today = close_pv.at[d, sym]
            high_today = high_pv.at[d, sym]
            if pd.isna(close_today):
                continue
            peak = max(peak, high_today)
            days_held = (d - e_date).days
            sl_px = peak * 0.92  # -8% trail
            target = e_px * 1.20
            exit_now = False
            exit_px = close_today
            reason = ""
            if low_pv.at[d, sym] <= sl_px and peak > e_px * 1.02:
                exit_now = True
                exit_px = sl_px
                reason = "trail"
            elif high_today >= target:
                exit_now = True
                exit_px = target
                reason = "target"
            elif days_held >= 90:
                exit_now = True
                exit_px = close_today
                reason = "time"
            elif low_pv.at[d, sym] <= e_px * 0.92 and peak < e_px * 1.02:
                # initial 8% SL never activated to trail
                exit_now = True
                exit_px = e_px * 0.92
                reason = "initSL"
            if exit_now:
                trades.append(dict(entry_date=e_date, exit_date=d, symbol=sym,
                                   entry_px=e_px, exit_px=exit_px, slot=slot))
                active_slots[slot] = None
            else:
                active_slots[slot] = (sym, e_date, e_px, peak, days_held)

        # entries
        if any(s is None for s in active_slots):
            for sym in close_pv.columns:
                c_today = close_pv.at[d, sym]
                if pd.isna(c_today):
                    continue
                prior_high = hh250.at[d, sym]
                if pd.isna(prior_high):
                    continue
                v_today = vol_pv.at[d, sym]
                v_avg = vol_sma20.at[d, sym]
                if pd.isna(v_avg) or v_avg == 0:
                    continue
                # already holding this symbol?
                already = any(s is not None and s[0] == sym for s in active_slots)
                if already:
                    continue
                if c_today > prior_high and v_today > 1.3 * v_avg:
                    # take open of next day
                    if i + 1 >= len(TRADE_DATES):
                        continue
                    nd = TRADE_DATES[i + 1]
                    epx = open_pv.at[nd, sym]
                    if pd.isna(epx):
                        continue
                    for slot in range(MAX_SLOTS):
                        if active_slots[slot] is None:
                            active_slots[slot] = (sym, nd, epx, epx, 0)
                            break
                if all(s is not None for s in active_slots):
                    break

    # close any open
    last_d = TRADE_DATES[-1]
    for slot in range(MAX_SLOTS):
        if active_slots[slot] is None:
            continue
        sym, e_date, e_px, peak, dh = active_slots[slot]
        exit_px = close_pv.at[last_d, sym]
        if pd.isna(exit_px):
            exit_px = close_pv[sym].dropna().iloc[-1]
        trades.append(dict(entry_date=e_date, exit_date=last_d, symbol=sym,
                           entry_px=e_px, exit_px=exit_px, slot=slot))
    return trades


# ===================================================================
# Model 3: Bollinger squeeze breakout
# ===================================================================
def model_bbsqueeze():
    trades = []
    active_slots = [None] * MAX_SLOTS  # (sym,e_date,e_px,days_held)
    ma20 = close_pv.rolling(20).mean()
    std20 = close_pv.rolling(20).std()
    upper = ma20 + 2 * std20
    width = (upper - (ma20 - 2 * std20)) / ma20
    # rolling 20th pct over last 60d
    width_q20 = width.rolling(60).quantile(0.20)
    prior_high = high_pv.shift(1)

    for i in range(len(TRADE_DATES)):
        d = TRADE_DATES[i]
        # exits
        for slot in range(MAX_SLOTS):
            if active_slots[slot] is None:
                continue
            sym, e_date, e_px, _ = active_slots[slot]
            c = close_pv.at[d, sym]
            lo = low_pv.at[d, sym]
            hi = high_pv.at[d, sym]
            if pd.isna(c):
                continue
            days_held = (d - e_date).days
            sl_px = e_px * 0.95
            target = e_px * 1.15
            exit_now = False
            exit_px = c
            if lo <= sl_px:
                exit_now = True
                exit_px = sl_px
            elif hi >= target:
                exit_now = True
                exit_px = target
            elif days_held >= 60:
                exit_now = True
                exit_px = c
            if exit_now:
                trades.append(dict(entry_date=e_date, exit_date=d, symbol=sym,
                                   entry_px=e_px, exit_px=exit_px, slot=slot))
                active_slots[slot] = None
            else:
                active_slots[slot] = (sym, e_date, e_px, days_held)

        # entries
        if any(s is None for s in active_slots):
            for sym in close_pv.columns:
                if any(s is not None and s[0] == sym for s in active_slots):
                    continue
                w = width.at[d, sym]
                wq = width_q20.at[d, sym]
                ph = prior_high.at[d, sym]
                u = upper.at[d, sym]
                c = close_pv.at[d, sym]
                if any(pd.isna(x) for x in (w, wq, ph, u, c)):
                    continue
                if w <= wq and c > u and c > ph:
                    if i + 1 >= len(TRADE_DATES):
                        continue
                    nd = TRADE_DATES[i + 1]
                    epx = open_pv.at[nd, sym]
                    if pd.isna(epx):
                        continue
                    for slot in range(MAX_SLOTS):
                        if active_slots[slot] is None:
                            active_slots[slot] = (sym, nd, epx, 0)
                            break
                if all(s is not None for s in active_slots):
                    break

    last_d = TRADE_DATES[-1]
    for slot in range(MAX_SLOTS):
        if active_slots[slot] is None:
            continue
        sym, e_date, e_px, _ = active_slots[slot]
        exit_px = close_pv.at[last_d, sym]
        if pd.isna(exit_px):
            exit_px = close_pv[sym].dropna().iloc[-1]
        trades.append(dict(entry_date=e_date, exit_date=last_d, symbol=sym,
                           entry_px=e_px, exit_px=exit_px, slot=slot))
    return trades


# ===================================================================
# Model 4: Multi-timeframe trend (daily > 50/200 ema, weekly > 50 ema)
# 3-month hold, quarterly rebalance, top-5 by largest 50/200 ratio
# ===================================================================
def model_mtf_trend():
    trades = []
    ema50 = close_pv.ewm(span=50, adjust=False).mean()
    ema200 = close_pv.ewm(span=200, adjust=False).mean()
    # weekly ema50: resample weekly Friday then ewm
    weekly = close_pv.resample("W-FRI").last()
    wema50 = weekly.ewm(span=50, adjust=False).mean()

    for i, rebal_d in enumerate(QUARTER_DAYS):
        # find the weekly bar <= rebal_d
        wk_idx = weekly.index[weekly.index <= rebal_d]
        if len(wk_idx) == 0:
            continue
        wk_d = wk_idx[-1]
        candidates = []
        for sym in close_pv.columns:
            c = close_pv.at[rebal_d, sym]
            e50 = ema50.at[rebal_d, sym]
            e200 = ema200.at[rebal_d, sym]
            wc = weekly.at[wk_d, sym] if sym in weekly.columns else np.nan
            we50 = wema50.at[wk_d, sym] if sym in wema50.columns else np.nan
            if any(pd.isna(x) for x in (c, e50, e200, wc, we50)):
                continue
            if c > e50 > e200 and wc > we50:
                candidates.append((sym, e50 / e200))  # rank by trend strength
        candidates.sort(key=lambda x: -x[1])
        top_syms = [s for s, _ in candidates[:MAX_SLOTS]]
        if not top_syms:
            continue
        if i + 1 < len(QUARTER_DAYS):
            exit_d = QUARTER_DAYS[i + 1]
        else:
            exit_d = TRADE_DATES[-1]
        for slot, sym in enumerate(top_syms):
            entry_idx = ALL_DATES.index(rebal_d) + 1
            if entry_idx >= len(ALL_DATES):
                continue
            entry_d = ALL_DATES[entry_idx]
            entry_px = open_pv.at[entry_d, sym]
            if pd.isna(entry_px):
                continue
            exit_idx_d = exit_d if exit_d in ALL_DATES else TRADE_DATES[-1]
            exit_px = open_pv.at[exit_idx_d, sym]
            if pd.isna(exit_px):
                exit_px = close_pv[sym].loc[:exit_idx_d].dropna().iloc[-1]
            trades.append(dict(entry_date=entry_d, exit_date=exit_idx_d, symbol=sym,
                               entry_px=entry_px, exit_px=exit_px, slot=slot))
    return trades


# -------------------- Run --------------------
M3_YEARLY = {2023: 80.87, 2024: 133.78, 2025: 46.14}

results = []
for name, fn in [
    ("M1_weekly_momentum_q", model_weekly_momentum),
    ("M2_52w_breakout_pos", model_52w_breakout),
    ("M3_bb_squeeze", model_bbsqueeze),
    ("M4_mtf_trend", model_mtf_trend),
]:
    print(f"\nBuilding trades for {name}...", flush=True)
    trades = fn()
    print(f"  {len(trades)} trades built", flush=True)
    eq = run_portfolio(trades, TRADE_DATES)
    r = summarize(name, trades, eq, M3_YEARLY)
    if r:
        r["equity"] = eq
        results.append(r)

# -------------------- Comparison table --------------------
print("\n\n" + "=" * 90)
print("COMPARISON TABLE")
print("=" * 90)
hdr = f"{'Model':<28}{'2023':>10}{'2024':>10}{'2025':>10}{'2026':>10}{'CAGR':>10}{'MaxDD':>10}{'Sharpe':>8}{'M3corr':>8}"
print(hdr)
print("-" * len(hdr))
for r in results:
    yrs = r["yearly"]
    yc = {y: yrs.get(y, float("nan")) for y in [2023, 2024, 2025, 2026]}
    print(f"{r['name']:<28}{yc[2023]:>10.2f}{yc[2024]:>10.2f}{yc[2025]:>10.2f}{yc[2026]:>10.2f}"
          f"{r['cagr']:>10.2f}{r['mdd']:>10.2f}{r['sharpe']:>8.2f}{r['m3_corr']:>8.3f}")
print("-" * len(hdr))
print(f"{'M3_reference':<28}{80.87:>10.2f}{133.78:>10.2f}{46.14:>10.2f}{'-':>10}{'87.0':>10}{'~-6':>10}{'-':>8}{'1.000':>8}")

# Save concise results
with open("/tmp/positional_results.json", "w") as f:
    json.dump([{k: (v if not isinstance(v, (pd.Series,)) else None) for k, v in r.items()}
               for r in results], f, default=str, indent=2)
print("\nDone. Saved /tmp/positional_results.json")
