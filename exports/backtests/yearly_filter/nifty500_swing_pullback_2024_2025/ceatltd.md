# Ceat Ltd. (CEATLTD)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 3266.00
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
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / Stop hits / Partials:** 0 / 5 / 1
- **Avg / median % per leg:** -0.79% / 1.59%
- **Sum % (uncompounded):** -4.75%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 0 | 5 | 1 | -0.79% | -4.7% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 0 | 5 | 1 | -0.79% | -4.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 0 | 5 | 1 | -0.79% | -4.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-06-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 00:00:00 | 2599.00 | 2450.23 | 2485.79 | Stage2 pullback-breakout RSI=64 vol=5.5x ATR=85.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-28 00:00:00 | 2769.92 | 2453.93 | 2517.80 | T1 booked 50% @ 2769.92 |
| Stop hit — per-position SL triggered | 2024-07-11 00:00:00 | 2640.25 | 2475.12 | 2620.42 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-08-09 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 00:00:00 | 2784.30 | 2508.76 | 2656.13 | Stage2 pullback-breakout RSI=59 vol=2.2x ATR=107.11 |
| Stop hit — per-position SL triggered | 2024-08-13 00:00:00 | 2623.63 | 2511.93 | 2657.96 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2024-08-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 00:00:00 | 2825.75 | 2521.50 | 2681.71 | Stage2 pullback-breakout RSI=60 vol=2.1x ATR=101.59 |
| Stop hit — per-position SL triggered | 2024-09-04 00:00:00 | 2898.00 | 2551.44 | 2779.05 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-09-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 00:00:00 | 3002.80 | 2573.91 | 2834.63 | Stage2 pullback-breakout RSI=67 vol=2.1x ATR=94.45 |
| Stop hit — per-position SL triggered | 2024-09-19 00:00:00 | 2861.12 | 2587.78 | 2863.30 | SL hit (bars_held=4) |

### Cycle 5 — BUY (started 2024-09-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-25 00:00:00 | 3111.95 | 2602.99 | 2903.50 | Stage2 pullback-breakout RSI=67 vol=2.9x ATR=103.27 |
| Stop hit — per-position SL triggered | 2024-10-07 00:00:00 | 2957.05 | 2638.01 | 3005.65 | SL hit (bars_held=7) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-06-27 00:00:00 | 2599.00 | 2024-06-28 00:00:00 | 2769.92 | PARTIAL | 0.50 | 6.58% |
| BUY | retest1 | 2024-06-27 00:00:00 | 2599.00 | 2024-07-11 00:00:00 | 2640.25 | STOP_HIT | 0.50 | 1.59% |
| BUY | retest1 | 2024-08-09 00:00:00 | 2784.30 | 2024-08-13 00:00:00 | 2623.63 | STOP_HIT | 1.00 | -5.77% |
| BUY | retest1 | 2024-08-21 00:00:00 | 2825.75 | 2024-09-04 00:00:00 | 2898.00 | STOP_HIT | 1.00 | 2.56% |
| BUY | retest1 | 2024-09-13 00:00:00 | 3002.80 | 2024-09-19 00:00:00 | 2861.12 | STOP_HIT | 1.00 | -4.72% |
| BUY | retest1 | 2024-09-25 00:00:00 | 3111.95 | 2024-10-07 00:00:00 | 2957.05 | STOP_HIT | 1.00 | -4.98% |
