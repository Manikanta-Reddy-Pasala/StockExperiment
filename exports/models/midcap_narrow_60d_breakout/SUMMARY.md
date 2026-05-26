# midcap_narrow_60d_breakout — SUMMARY

**Mid+Small breakout swing. Top-100 ADV from N500 MINUS Large. 40d breakout, +100% target, -20% trail, 120d max-hold, no SMA20 exit.**

> **Live↔backtest aligned 2026-05-26.** `build_universe.py` previously emitted a
> different live universe (skip-top-30, keep-100, **no large-cap exclusion** → 100
> contaminated names incl TITAN/MARUTI/ITC/TRITURBINE) than this backtest uses, so
> live ran ~+83%/yr at 22% DD instead of the figures below. The builder now mirrors
> the backtest exactly (top-100 ADV minus Nifty-100 → ~42 midcaps). Backtest
> reconfirmed 2026-05-26 on the aligned universe: **+141.73% CAGR / 8.12% DD /
> Calmar 17.46 / 8 trades / 75% WR** (3yr to 2026-05-15; the +137.85% below is the
> same run to 2026-05-12).

## When it BUYS (entry rules)

Single position (`max_concurrent=1`). When **flat**, every run scans the universe (~42 midcaps
= top-100 ADV from N500 minus Nifty-100) for a **fresh breakout** and buys the strongest:
- **40-day high** — today's close is above the highest high of the prior 40 days (`HH_WINDOW=40`).
- **Volume surge** — today's volume **≥ 2× the 20-day avg volume** (`VOL_MULT=2.0`).
- **Stage-2 trend** — close **> 200-day SMA** (`SMA_LONG=200`).
- All three must fire the same day. If several stocks qualify, it picks the one with the
  **highest volume ratio** (`vol_ratio`). Backtest enters next-day at the open; live enters at
  the breakout close.
- Code: entry RULE = shared `tools/shared/breakout_strategy.is_breakout` — called by BOTH `scan_entry_candidate()` (live) and the backtest scan loop. SELECTION (universe) is per-model. Parity-tested.

> Name is "60d" for legacy reasons (v1 used a 60-day high); the live/v2 logic uses a **40-day**
> window. Not renamed because the name is the DB key (model_settings / model_ledger / model_trades).

## When it SELLS (exit rules)

Breakout swing, single position, checked every run (09:25 + 15:25). Unlike the rotation
models, it does **NOT** sell to chase a new breakout — it rides each position until one of
these fires (first wins). Code: exit RULE = shared `tools/shared/breakout_strategy.breakout_exit_reason`
— called by BOTH `check_exit()` (live) and the backtest loop, so they can't drift:

| Reason | Fires when | Constant |
|---|---|---|
| **TARGET** | current close **≥ +100%** above entry | `TARGET_PCT=1.00` |
| **STOP** | current close **≤ −20%** below entry (catastrophe stop, added 2026-05-26; fires rarely, caps unbounded downside) | `STOP_PCT=0.20` |
| **TRAIL** | trade is **≥ +10% in profit** AND current close is **≥ 20% below the peak close** since entry | `PROFIT_TRIGGER=0.10`, `TRAIL_PCT=0.20` |
| **MAX_HOLD** | **120 calendar days** held → force-exit at market | `MAX_HOLD_DAYS=120` |
| SMA20 | **disabled** (leaked winners on dips) | `USE_SMA_EXIT=False` |

So a position is held until target/stop/trail/max-hold — a fresh breakout elsewhere is ignored
until the current one exits.

### How TRAIL calculates (the part that confuses people)

The trail is **20% off the PEAK PRICE**, not a 20% drop in the gain-number. Each run it
tracks `peak = highest close since entry`, then:

```
ret_entry = (close − entry) / entry      # gain vs entry
ret_peak  = (peak − close) / peak         # drop vs peak
TRAIL fires when  ret_entry ≥ +10%  AND  ret_peak ≥ 20%
```

**Worked example — peak hit +40%:**
- Peak price = 1.40 × entry. 20% below that peak = 1.40 × 0.80 = **1.12 = +12% from entry**.
- So TRAIL fires when price falls back to **+12%**, NOT at +30%.
- At +30% (price 1.30) the drop-from-peak is only (1.40−1.30)/1.40 = **7.1% < 20% → no exit**.

The `+10%` arm and the `20%-off-peak` test must both hold at once. Below +10% the TRAIL never
fires — the **STOP** (−20% from entry) handles deep losers instead.

## Backtest window & trade frequency

| Metric | Value |
|---|---|
| Backtest window | **2023-05-15 → 2026-05-12** (≈3.00 years) |
| First entry | 2023-05-18 |
| Last exit | 2026-01-21 |
| Total trades | 8 |
| Trades per year | ~2.7 |
| Rebalance | Event-driven |
| Data source | **Fyers** (498/504 N500, 4-yr re-pull, cont_flag=1) |

## Headline result

| Metric | Value |
|---|---:|
| Final NAV | **₹13,456,535** |
| Total return | **+1245.65%** |
| **3-yr CAGR** | **+137.85%/yr** |
| Max DD (cash NAV) | 8.12% |
| Calmar | 16.98 |
| Trades | 8 |
| WR | 75.0% (6W / 2L) |

## Yearly money flow

| Year | Open | Close | ROI | Trades |
|---|---:|---:|---:|---:|
| 2023-24 | ₹1,000,000 | ₹1,570,706 | **+57.07%** | 2 |
| 2024-25 | ₹1,570,706 | ₹7,368,992 | **+369.15%** | 3 |
| 2025-26 | ₹7,368,992 | ₹13,456,535 | **+82.61%** | 3 |

## Returns by NSE cap segment

| Cap | Trades | Wins | Losses | WR | Total PnL ₹ |
|---|---:|---:|---:|---:|---:|
| **Mid** | 3 | 3 | 0 | 100% | +4,267,954 |
| **Small** | 5 | 3 | 2 | 60% | +8,188,741 |