# Amara Raja Energy & Mobility Ltd. (ARE&M)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 876.55
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
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 6 / 2
- **Target hits / Stop hits / Partials:** 2 / 2 / 4
- **Avg / median % per leg:** 4.70% / 3.68%
- **Sum % (uncompounded):** 37.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 6 | 75.0% | 2 | 2 | 4 | 4.70% | 37.6% |
| BUY @ 2nd Alert (retest1) | 8 | 6 | 75.0% | 2 | 2 | 4 | 4.70% | 37.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 6 | 75.0% | 2 | 2 | 4 | 4.70% | 37.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 00:00:00 | 649.95 | 609.53 | 629.49 | Stage2 pullback-breakout RSI=65 vol=3.2x ATR=10.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-08 00:00:00 | 671.16 | 610.96 | 636.86 | T1 booked 50% @ 671.16 |
| Stop hit — per-position SL triggered | 2023-09-12 00:00:00 | 649.95 | 611.87 | 640.50 | SL hit (bars_held=5) |

### Cycle 2 — BUY (started 2023-11-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-21 00:00:00 | 655.05 | 620.71 | 635.96 | Stage2 pullback-breakout RSI=64 vol=2.8x ATR=12.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-24 00:00:00 | 679.12 | 622.03 | 643.84 | T1 booked 50% @ 679.12 |
| Target hit | 2024-01-17 00:00:00 | 796.95 | 671.00 | 797.93 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-01-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-29 00:00:00 | 837.95 | 680.48 | 805.08 | Stage2 pullback-breakout RSI=63 vol=1.6x ATR=23.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-31 00:00:00 | 884.47 | 684.10 | 815.93 | T1 booked 50% @ 884.47 |
| Target hit | 2024-02-12 00:00:00 | 845.45 | 699.08 | 849.30 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-03-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-07 00:00:00 | 869.35 | 725.01 | 849.08 | Stage2 pullback-breakout RSI=61 vol=2.1x ATR=23.13 |
| Stop hit — per-position SL triggered | 2024-03-13 00:00:00 | 834.66 | 728.26 | 844.66 | SL hit (bars_held=3) |

### Cycle 5 — BUY (started 2024-04-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-08 00:00:00 | 869.05 | 737.31 | 806.86 | Stage2 pullback-breakout RSI=65 vol=5.6x ATR=28.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-16 00:00:00 | 926.26 | 745.33 | 845.30 | T1 booked 50% @ 926.26 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-05 00:00:00 | 649.95 | 2023-09-08 00:00:00 | 671.16 | PARTIAL | 0.50 | 3.26% |
| BUY | retest1 | 2023-09-05 00:00:00 | 649.95 | 2023-09-12 00:00:00 | 649.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-21 00:00:00 | 655.05 | 2023-11-24 00:00:00 | 679.12 | PARTIAL | 0.50 | 3.68% |
| BUY | retest1 | 2023-11-21 00:00:00 | 655.05 | 2024-01-17 00:00:00 | 796.95 | TARGET_HIT | 0.50 | 21.66% |
| BUY | retest1 | 2024-01-29 00:00:00 | 837.95 | 2024-01-31 00:00:00 | 884.47 | PARTIAL | 0.50 | 5.55% |
| BUY | retest1 | 2024-01-29 00:00:00 | 837.95 | 2024-02-12 00:00:00 | 845.45 | TARGET_HIT | 0.50 | 0.90% |
| BUY | retest1 | 2024-03-07 00:00:00 | 869.35 | 2024-03-13 00:00:00 | 834.66 | STOP_HIT | 1.00 | -3.99% |
| BUY | retest1 | 2024-04-08 00:00:00 | 869.05 | 2024-04-16 00:00:00 | 926.26 | PARTIAL | 0.50 | 6.58% |
