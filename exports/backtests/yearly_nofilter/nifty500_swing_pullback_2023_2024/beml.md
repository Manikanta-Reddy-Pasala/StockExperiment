# BEML Ltd. (BEML)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1894.10
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
- **Avg / median % per leg:** 8.99% / 8.14%
- **Sum % (uncompounded):** 53.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 2 | 2 | 2 | 8.99% | 54.0% |
| BUY @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 2 | 2 | 2 | 8.99% | 54.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 4 | 66.7% | 2 | 2 | 2 | 8.99% | 54.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-21 00:00:00 | 841.45 | 743.73 | 796.06 | Stage2 pullback-breakout RSI=66 vol=4.1x ATR=25.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-24 00:00:00 | 892.07 | 745.14 | 804.65 | T1 booked 50% @ 892.07 |
| Target hit | 2023-09-12 00:00:00 | 1155.97 | 846.19 | 1163.43 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-10-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-16 00:00:00 | 1208.05 | 910.05 | 1167.14 | Stage2 pullback-breakout RSI=60 vol=2.5x ATR=39.79 |
| Stop hit — per-position SL triggered | 2023-10-20 00:00:00 | 1148.36 | 920.10 | 1166.38 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2023-12-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-22 00:00:00 | 1402.08 | 1009.43 | 1265.43 | Stage2 pullback-breakout RSI=69 vol=2.2x ATR=57.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-05 00:00:00 | 1516.26 | 1046.09 | 1369.43 | T1 booked 50% @ 1516.26 |
| Target hit | 2024-02-09 00:00:00 | 1623.58 | 1178.67 | 1685.79 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-02-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-27 00:00:00 | 1686.38 | 1217.92 | 1582.95 | Stage2 pullback-breakout RSI=58 vol=2.8x ATR=94.85 |
| Stop hit — per-position SL triggered | 2024-02-29 00:00:00 | 1544.10 | 1225.28 | 1583.86 | SL hit (bars_held=2) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-21 00:00:00 | 841.45 | 2023-07-24 00:00:00 | 892.07 | PARTIAL | 0.50 | 6.02% |
| BUY | retest1 | 2023-07-21 00:00:00 | 841.45 | 2023-09-12 00:00:00 | 1155.97 | TARGET_HIT | 0.50 | 37.38% |
| BUY | retest1 | 2023-10-16 00:00:00 | 1208.05 | 2023-10-20 00:00:00 | 1148.36 | STOP_HIT | 1.00 | -4.94% |
| BUY | retest1 | 2023-12-22 00:00:00 | 1402.08 | 2024-01-05 00:00:00 | 1516.26 | PARTIAL | 0.50 | 8.14% |
| BUY | retest1 | 2023-12-22 00:00:00 | 1402.08 | 2024-02-09 00:00:00 | 1623.58 | TARGET_HIT | 0.50 | 15.80% |
| BUY | retest1 | 2024-02-27 00:00:00 | 1686.38 | 2024-02-29 00:00:00 | 1544.10 | STOP_HIT | 1.00 | -8.44% |
