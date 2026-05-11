# SBIN (SBIN)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1018.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 0 |
| ALERT2 | 0 |
| ALERT2_SKIP | 0 |
| ALERT3 | 19 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 21 |
| PARTIAL | 0 |
| TARGET_HIT | 14 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 20 / 1
- **Target hits / Stop hits / Partials:** 12 / 9 / 0
- **Avg / median % per leg:** 5.00% / 6.94%
- **Sum % (uncompounded):** 104.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 20 | 95.2% | 12 | 9 | 0 | 5.00% | 105.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 21 | 20 | 95.2% | 12 | 9 | 0 | 5.00% | 105.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 21 | 20 | 95.2% | 12 | 9 | 0 | 5.00% | 105.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 10:15:00 | 1017.35 | 1070.19 | 1070.41 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 1107.60 | 1069.70 | 1069.65 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-12 09:15:00 | 799.50 | 2025-08-07 10:15:00 | 798.60 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-05-21 12:30:00 | 783.85 | 2025-08-29 09:15:00 | 799.15 | STOP_HIT | 1.00 | 1.95% |
| BUY | retest2 | 2025-06-20 11:15:00 | 798.05 | 2025-08-29 13:15:00 | 801.75 | STOP_HIT | 1.00 | 0.46% |
| BUY | retest2 | 2025-06-20 15:00:00 | 797.00 | 2025-08-29 13:15:00 | 801.75 | STOP_HIT | 1.00 | 0.60% |
| BUY | retest2 | 2025-06-24 09:15:00 | 801.00 | 2025-08-29 13:15:00 | 801.75 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2025-06-25 09:15:00 | 797.05 | 2025-09-05 09:15:00 | 807.50 | STOP_HIT | 1.00 | 1.31% |
| BUY | retest2 | 2025-08-06 12:15:00 | 804.20 | 2025-09-09 09:15:00 | 806.50 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2025-08-07 14:30:00 | 804.45 | 2025-09-09 09:15:00 | 806.50 | STOP_HIT | 1.00 | 0.25% |
| BUY | retest2 | 2025-08-08 14:00:00 | 805.25 | 2025-09-09 09:15:00 | 806.50 | STOP_HIT | 1.00 | 0.16% |
| BUY | retest2 | 2025-08-08 14:30:00 | 806.00 | 2025-09-19 10:15:00 | 861.96 | TARGET_HIT | 1.00 | 6.94% |
| BUY | retest2 | 2025-08-11 09:15:00 | 818.00 | 2025-09-19 10:15:00 | 860.86 | TARGET_HIT | 1.00 | 5.24% |
| BUY | retest2 | 2025-08-29 11:30:00 | 806.10 | 2025-09-19 10:15:00 | 862.24 | TARGET_HIT | 1.00 | 6.96% |
| BUY | retest2 | 2025-08-29 12:00:00 | 805.50 | 2025-09-24 09:15:00 | 879.45 | TARGET_HIT | 1.00 | 9.18% |
| BUY | retest2 | 2025-08-29 12:30:00 | 805.40 | 2025-09-24 09:15:00 | 877.86 | TARGET_HIT | 1.00 | 9.00% |
| BUY | retest2 | 2025-09-05 09:15:00 | 811.10 | 2025-09-24 09:15:00 | 876.70 | TARGET_HIT | 1.00 | 8.09% |
| BUY | retest2 | 2025-09-08 09:15:00 | 810.80 | 2025-09-24 09:15:00 | 876.75 | TARGET_HIT | 1.00 | 8.13% |
| BUY | retest2 | 2025-09-08 10:30:00 | 811.10 | 2025-10-10 10:15:00 | 881.10 | TARGET_HIT | 1.00 | 8.63% |
| BUY | retest2 | 2025-09-08 12:00:00 | 810.30 | 2025-10-13 09:15:00 | 884.90 | TARGET_HIT | 1.00 | 9.21% |
| BUY | retest2 | 2025-09-09 11:30:00 | 808.60 | 2025-10-13 09:15:00 | 885.78 | TARGET_HIT | 1.00 | 9.54% |
| BUY | retest2 | 2025-09-09 14:00:00 | 809.00 | 2025-10-13 10:15:00 | 886.60 | TARGET_HIT | 1.00 | 9.59% |
| BUY | retest2 | 2025-09-10 09:15:00 | 812.70 | 2025-10-16 09:15:00 | 889.46 | TARGET_HIT | 1.00 | 9.45% |
