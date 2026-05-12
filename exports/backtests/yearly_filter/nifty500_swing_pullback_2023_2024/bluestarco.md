# Blue Star Ltd. (BLUESTARCO)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 1692.80
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
| PARTIAL | 3 |
| TARGET_HIT | 3 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 1
- **Target hits / Stop hits / Partials:** 3 / 1 / 3
- **Avg / median % per leg:** 8.01% / 6.65%
- **Sum % (uncompounded):** 56.09%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 6 | 85.7% | 3 | 1 | 3 | 8.01% | 56.1% |
| BUY @ 2nd Alert (retest1) | 7 | 6 | 85.7% | 3 | 1 | 3 | 8.01% | 56.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 6 | 85.7% | 3 | 1 | 3 | 8.01% | 56.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-07 00:00:00 | 793.05 | 709.00 | 748.42 | Stage2 pullback-breakout RSI=68 vol=3.5x ATR=17.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-20 00:00:00 | 828.98 | 716.81 | 785.16 | T1 booked 50% @ 828.98 |
| Target hit | 2023-10-23 00:00:00 | 883.00 | 751.53 | 885.19 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-11-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-03 00:00:00 | 971.70 | 763.05 | 897.77 | Stage2 pullback-breakout RSI=68 vol=1.9x ATR=39.17 |
| Stop hit — per-position SL triggered | 2023-11-09 00:00:00 | 912.94 | 770.28 | 913.59 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2024-01-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-11 00:00:00 | 996.00 | 840.31 | 960.94 | Stage2 pullback-breakout RSI=63 vol=2.4x ATR=26.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-15 00:00:00 | 1049.19 | 844.31 | 975.88 | T1 booked 50% @ 1049.19 |
| Target hit | 2024-03-13 00:00:00 | 1266.80 | 968.09 | 1269.98 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-03-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-27 00:00:00 | 1301.00 | 993.69 | 1267.53 | Stage2 pullback-breakout RSI=58 vol=1.6x ATR=46.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-08 00:00:00 | 1393.22 | 1016.67 | 1304.71 | T1 booked 50% @ 1393.22 |
| Target hit | 2024-05-09 00:00:00 | 1387.50 | 1092.41 | 1421.15 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-07 00:00:00 | 793.05 | 2023-09-20 00:00:00 | 828.98 | PARTIAL | 0.50 | 4.53% |
| BUY | retest1 | 2023-09-07 00:00:00 | 793.05 | 2023-10-23 00:00:00 | 883.00 | TARGET_HIT | 0.50 | 11.34% |
| BUY | retest1 | 2023-11-03 00:00:00 | 971.70 | 2023-11-09 00:00:00 | 912.94 | STOP_HIT | 1.00 | -6.05% |
| BUY | retest1 | 2024-01-11 00:00:00 | 996.00 | 2024-01-15 00:00:00 | 1049.19 | PARTIAL | 0.50 | 5.34% |
| BUY | retest1 | 2024-01-11 00:00:00 | 996.00 | 2024-03-13 00:00:00 | 1266.80 | TARGET_HIT | 0.50 | 27.19% |
| BUY | retest1 | 2024-03-27 00:00:00 | 1301.00 | 2024-04-08 00:00:00 | 1393.22 | PARTIAL | 0.50 | 7.09% |
| BUY | retest1 | 2024-03-27 00:00:00 | 1301.00 | 2024-05-09 00:00:00 | 1387.50 | TARGET_HIT | 0.50 | 6.65% |
