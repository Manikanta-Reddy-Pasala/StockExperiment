# HDFCBANK (HDFCBANK)

## Backtest Summary

- **Window:** 2024-10-07 09:15:00 → 2026-05-08 15:15:00 (2741 bars)
- **Last close:** 781.25
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 1 |
| ALERT3 | 10 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 17 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 13
- **Target hits / Stop hits / Partials:** 2 / 15 / 2
- **Avg / median % per leg:** 0.88% / -0.46%
- **Sum % (uncompounded):** 16.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 2 | 13.3% | 0 | 15 | 0 | -0.89% | -13.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 15 | 2 | 13.3% | 0 | 15 | 0 | -0.89% | -13.3% |
| SELL (all) | 4 | 4 | 100.0% | 2 | 0 | 2 | 7.50% | 30.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 4 | 100.0% | 2 | 0 | 2 | 7.50% | 30.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 19 | 6 | 31.6% | 2 | 15 | 2 | 0.88% | 16.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 12:15:00 | 948.60 | 989.05 | 989.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 09:15:00 | 945.70 | 987.46 | 988.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 947.60 | 946.81 | 961.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 947.60 | 946.81 | 961.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 947.60 | 946.81 | 961.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 09:15:00 | 944.90 | 947.58 | 960.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 10:00:00 | 944.15 | 947.54 | 960.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 15:15:00 | 897.65 | 929.26 | 944.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 15:15:00 | 896.94 | 929.26 | 944.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 850.41 | 922.93 | 940.14 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 849.74 | 922.93 | 940.14 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-11-11 13:15:00 | 987.95 | 2025-11-14 09:15:00 | 980.05 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-11-11 14:00:00 | 988.20 | 2025-11-14 09:15:00 | 980.05 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-11-14 15:00:00 | 990.45 | 2025-12-17 10:15:00 | 987.20 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-11-19 10:00:00 | 986.90 | 2025-12-17 10:15:00 | 987.20 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2025-11-19 11:45:00 | 986.65 | 2025-12-22 10:15:00 | 987.30 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2025-12-10 13:15:00 | 989.70 | 2025-12-30 11:15:00 | 985.00 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-12-10 15:15:00 | 991.70 | 2025-12-31 09:15:00 | 989.60 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-12-11 10:00:00 | 994.40 | 2026-01-05 10:15:00 | 988.00 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-12-22 09:15:00 | 992.60 | 2026-01-05 10:15:00 | 988.00 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-12-23 09:45:00 | 991.40 | 2026-01-05 10:15:00 | 988.00 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2025-12-30 15:15:00 | 995.00 | 2026-01-05 10:15:00 | 988.00 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-12-31 12:30:00 | 993.40 | 2026-01-05 13:15:00 | 978.15 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-12-31 15:15:00 | 994.00 | 2026-01-05 13:15:00 | 978.15 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2026-01-02 09:15:00 | 994.60 | 2026-01-05 13:15:00 | 978.15 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2026-01-02 15:00:00 | 1001.95 | 2026-01-06 09:15:00 | 963.25 | STOP_HIT | 1.00 | -3.86% |
| SELL | retest2 | 2026-02-06 09:15:00 | 944.90 | 2026-02-26 15:15:00 | 897.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-06 10:00:00 | 944.15 | 2026-02-26 15:15:00 | 896.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-06 09:15:00 | 944.90 | 2026-03-04 09:15:00 | 850.41 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-06 10:00:00 | 944.15 | 2026-03-04 09:15:00 | 849.74 | TARGET_HIT | 0.50 | 10.00% |
