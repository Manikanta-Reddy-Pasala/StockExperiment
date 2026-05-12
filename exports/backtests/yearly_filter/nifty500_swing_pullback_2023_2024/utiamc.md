# UTI Asset Management Company Ltd. (UTIAMC)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 966.05
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
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 2
- **Target hits / Stop hits / Partials:** 1 / 3 / 2
- **Avg / median % per leg:** 2.01% / 3.10%
- **Sum % (uncompounded):** 12.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 1 | 3 | 2 | 2.01% | 12.1% |
| BUY @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 1 | 3 | 2 | 2.01% | 12.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 4 | 66.7% | 1 | 3 | 2 | 2.01% | 12.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 00:00:00 | 773.85 | 743.36 | 759.25 | Stage2 pullback-breakout RSI=56 vol=3.1x ATR=18.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-11 00:00:00 | 809.98 | 745.46 | 772.13 | T1 booked 50% @ 809.98 |
| Stop hit — per-position SL triggered | 2023-09-12 00:00:00 | 773.85 | 745.95 | 774.29 | SL hit (bars_held=5) |

### Cycle 2 — BUY (started 2023-10-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-13 00:00:00 | 818.85 | 754.05 | 789.13 | Stage2 pullback-breakout RSI=66 vol=6.2x ATR=19.15 |
| Stop hit — per-position SL triggered | 2023-10-19 00:00:00 | 790.12 | 755.87 | 792.67 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2024-01-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-30 00:00:00 | 896.30 | 793.81 | 865.16 | Stage2 pullback-breakout RSI=61 vol=12.3x ATR=30.29 |
| Stop hit — per-position SL triggered | 2024-02-13 00:00:00 | 912.35 | 805.71 | 898.02 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-04-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-04 00:00:00 | 895.75 | 824.79 | 852.81 | Stage2 pullback-breakout RSI=60 vol=1.6x ATR=26.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-22 00:00:00 | 949.53 | 833.36 | 891.73 | T1 booked 50% @ 949.53 |
| Target hit | 2024-05-06 00:00:00 | 923.55 | 843.36 | 925.27 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-05 00:00:00 | 773.85 | 2023-09-11 00:00:00 | 809.98 | PARTIAL | 0.50 | 4.67% |
| BUY | retest1 | 2023-09-05 00:00:00 | 773.85 | 2023-09-12 00:00:00 | 773.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-13 00:00:00 | 818.85 | 2023-10-19 00:00:00 | 790.12 | STOP_HIT | 1.00 | -3.51% |
| BUY | retest1 | 2024-01-30 00:00:00 | 896.30 | 2024-02-13 00:00:00 | 912.35 | STOP_HIT | 1.00 | 1.79% |
| BUY | retest1 | 2024-04-04 00:00:00 | 895.75 | 2024-04-22 00:00:00 | 949.53 | PARTIAL | 0.50 | 6.00% |
| BUY | retest1 | 2024-04-04 00:00:00 | 895.75 | 2024-05-06 00:00:00 | 923.55 | TARGET_HIT | 0.50 | 3.10% |
