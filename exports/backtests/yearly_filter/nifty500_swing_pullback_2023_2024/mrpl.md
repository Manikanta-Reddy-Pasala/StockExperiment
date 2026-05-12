# Mangalore Refinery & Petrochemicals Ltd. (MRPL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 159.80
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
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / Stop hits / Partials:** 0 / 4 / 1
- **Avg / median % per leg:** -0.71% / 0.00%
- **Sum % (uncompounded):** -3.54%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 0 | 4 | 1 | -0.71% | -3.5% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 0 | 4 | 1 | -0.71% | -3.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 2 | 40.0% | 0 | 4 | 1 | -0.71% | -3.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-21 00:00:00 | 91.90 | 69.05 | 85.08 | Stage2 pullback-breakout RSI=68 vol=5.2x ATR=3.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-01 00:00:00 | 98.34 | 71.17 | 90.46 | T1 booked 50% @ 98.34 |
| Stop hit — per-position SL triggered | 2023-09-12 00:00:00 | 91.90 | 72.80 | 92.74 | SL hit (bars_held=16) |

### Cycle 2 — BUY (started 2023-10-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-04 00:00:00 | 97.85 | 75.60 | 94.13 | Stage2 pullback-breakout RSI=65 vol=5.1x ATR=3.74 |
| Stop hit — per-position SL triggered | 2023-10-09 00:00:00 | 92.23 | 76.20 | 94.54 | SL hit (bars_held=3) |

### Cycle 3 — BUY (started 2023-11-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-03 00:00:00 | 109.90 | 80.56 | 101.85 | Stage2 pullback-breakout RSI=64 vol=2.2x ATR=5.95 |
| Stop hit — per-position SL triggered | 2023-11-22 00:00:00 | 113.15 | 84.70 | 111.37 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-04-24 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 00:00:00 | 248.70 | 157.31 | 227.55 | Stage2 pullback-breakout RSI=65 vol=6.9x ATR=12.87 |
| Stop hit — per-position SL triggered | 2024-05-06 00:00:00 | 229.39 | 163.44 | 237.40 | SL hit (bars_held=7) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-21 00:00:00 | 91.90 | 2023-09-01 00:00:00 | 98.34 | PARTIAL | 0.50 | 7.01% |
| BUY | retest1 | 2023-08-21 00:00:00 | 91.90 | 2023-09-12 00:00:00 | 91.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-04 00:00:00 | 97.85 | 2023-10-09 00:00:00 | 92.23 | STOP_HIT | 1.00 | -5.74% |
| BUY | retest1 | 2023-11-03 00:00:00 | 109.90 | 2023-11-22 00:00:00 | 113.15 | STOP_HIT | 1.00 | 2.96% |
| BUY | retest1 | 2024-04-24 00:00:00 | 248.70 | 2024-05-06 00:00:00 | 229.39 | STOP_HIT | 1.00 | -7.76% |
