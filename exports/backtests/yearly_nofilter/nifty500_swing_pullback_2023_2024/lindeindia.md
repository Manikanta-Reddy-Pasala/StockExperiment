# Linde India Ltd. (LINDEINDIA)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 7659.50
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
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 2
- **Target hits / Stop hits / Partials:** 2 / 2 / 2
- **Avg / median % per leg:** 10.13% / 5.46%
- **Sum % (uncompounded):** 60.75%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 2 | 2 | 2 | 10.13% | 60.8% |
| BUY @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 2 | 2 | 2 | 10.13% | 60.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 4 | 66.7% | 2 | 2 | 2 | 10.13% | 60.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-20 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-20 00:00:00 | 4593.95 | 3842.59 | 4381.49 | Stage2 pullback-breakout RSI=65 vol=3.2x ATR=107.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-25 00:00:00 | 4809.68 | 3865.15 | 4440.27 | T1 booked 50% @ 4809.68 |
| Target hit | 2023-09-15 00:00:00 | 5922.00 | 4417.41 | 6013.99 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-01-19 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-19 00:00:00 | 5822.25 | 5237.35 | 5603.31 | Stage2 pullback-breakout RSI=60 vol=7.1x ATR=155.88 |
| Stop hit — per-position SL triggered | 2024-01-31 00:00:00 | 5588.42 | 5267.63 | 5640.41 | SL hit (bars_held=7) |

### Cycle 3 — BUY (started 2024-02-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-16 00:00:00 | 5787.35 | 5305.49 | 5619.09 | Stage2 pullback-breakout RSI=59 vol=1.6x ATR=162.17 |
| Stop hit — per-position SL triggered | 2024-02-22 00:00:00 | 5544.09 | 5319.54 | 5631.35 | SL hit (bars_held=4) |

### Cycle 4 — BUY (started 2024-03-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-11 00:00:00 | 5906.80 | 5345.24 | 5587.98 | Stage2 pullback-breakout RSI=64 vol=5.1x ATR=161.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 00:00:00 | 6229.51 | 5354.94 | 5657.72 | T1 booked 50% @ 6229.51 |
| Target hit | 2024-05-08 00:00:00 | 7673.15 | 5927.53 | 7712.28 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-20 00:00:00 | 4593.95 | 2023-07-25 00:00:00 | 4809.68 | PARTIAL | 0.50 | 4.70% |
| BUY | retest1 | 2023-07-20 00:00:00 | 4593.95 | 2023-09-15 00:00:00 | 5922.00 | TARGET_HIT | 0.50 | 28.91% |
| BUY | retest1 | 2024-01-19 00:00:00 | 5822.25 | 2024-01-31 00:00:00 | 5588.42 | STOP_HIT | 1.00 | -4.02% |
| BUY | retest1 | 2024-02-16 00:00:00 | 5787.35 | 2024-02-22 00:00:00 | 5544.09 | STOP_HIT | 1.00 | -4.20% |
| BUY | retest1 | 2024-03-11 00:00:00 | 5906.80 | 2024-03-12 00:00:00 | 6229.51 | PARTIAL | 0.50 | 5.46% |
| BUY | retest1 | 2024-03-11 00:00:00 | 5906.80 | 2024-05-08 00:00:00 | 7673.15 | TARGET_HIT | 0.50 | 29.90% |
