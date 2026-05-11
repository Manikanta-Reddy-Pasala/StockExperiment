# Tata Elxsi Ltd. (TATAELXSI)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 4319.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 0 |
| ALERT1 | 0 |
| ALERT2 | 0 |
| ALERT2_SKIP | 0 |
| ALERT3 | 0 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 1
- **Avg / median % per leg:** 1.03% / -2.27%
- **Sum % (uncompounded):** 5.14%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | 1.03% | 5.1% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | 1.03% | 5.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 2 | 40.0% | 1 | 3 | 1 | 1.03% | 5.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-14 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-14 05:30:00 | 7730.75 | 7219.14 | 7558.43 | Stage2 pullback-breakout RSI=62 vol=2.3x ATR=139.97 |
| Stop hit — per-position SL triggered | 2023-07-18 05:30:00 | 7520.80 | 7227.29 | 7570.81 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2023-08-24 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-24 05:30:00 | 7278.65 | 7215.60 | 7162.77 | Stage2 pullback-breakout RSI=56 vol=2.4x ATR=109.97 |
| Stop hit — per-position SL triggered | 2023-08-31 05:30:00 | 7113.69 | 7221.73 | 7232.38 | SL hit (bars_held=5) |

### Cycle 3 — BUY (started 2023-11-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-06 05:30:00 | 7831.60 | 7288.55 | 7522.83 | Stage2 pullback-breakout RSI=65 vol=1.6x ATR=161.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-07 05:30:00 | 8154.35 | 7297.33 | 7584.58 | T1 booked 50% @ 8154.35 |
| Target hit | 2024-01-03 05:30:00 | 8607.55 | 7701.71 | 8685.49 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-04-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-03 05:30:00 | 7997.90 | 7759.54 | 7719.45 | Stage2 pullback-breakout RSI=61 vol=3.9x ATR=208.55 |
| Stop hit — per-position SL triggered | 2024-04-15 05:30:00 | 7685.07 | 7766.10 | 7774.27 | SL hit (bars_held=7) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-14 05:30:00 | 7730.75 | 2023-07-18 05:30:00 | 7520.80 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest1 | 2023-08-24 05:30:00 | 7278.65 | 2023-08-31 05:30:00 | 7113.69 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest1 | 2023-11-06 05:30:00 | 7831.60 | 2023-11-07 05:30:00 | 8154.35 | PARTIAL | 0.50 | 4.12% |
| BUY | retest1 | 2023-11-06 05:30:00 | 7831.60 | 2024-01-03 05:30:00 | 8607.55 | TARGET_HIT | 0.50 | 9.91% |
| BUY | retest1 | 2024-04-03 05:30:00 | 7997.90 | 2024-04-15 05:30:00 | 7685.07 | STOP_HIT | 1.00 | -3.91% |
