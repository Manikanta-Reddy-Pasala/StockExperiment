# CCL Products (I) Ltd. (CCL)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1122.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 18 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 19 |
| PARTIAL | 0 |
| TARGET_HIT | 6 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 16
- **Target hits / Stop hits / Partials:** 6 / 16 / 0
- **Avg / median % per leg:** 0.11% / -2.93%
- **Sum % (uncompounded):** 2.49%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 22 | 6 | 27.3% | 6 | 16 | 0 | 0.11% | 2.5% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -5.46% | -16.4% |
| BUY @ 3rd Alert (retest2) | 19 | 6 | 31.6% | 6 | 13 | 0 | 0.99% | 18.9% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -5.46% | -16.4% |
| retest2 (combined) | 19 | 6 | 31.6% | 6 | 13 | 0 | 0.99% | 18.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 09:15:00 | 837.85 | 865.64 | 865.68 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 11:15:00 | 988.15 | 864.26 | 863.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 09:15:00 | 1012.45 | 870.19 | 866.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 09:15:00 | 970.40 | 971.04 | 934.78 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-02 14:45:00 | 988.70 | 971.44 | 935.87 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-03 09:30:00 | 992.80 | 971.84 | 936.43 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 11:15:00 | 990.90 | 973.71 | 938.77 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 936.70 | 973.30 | 939.94 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-05 11:15:00 | 936.70 | 973.30 | 939.94 | SL hit (close<ema400) qty=1.00 sl=939.94 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-08-08 15:15:00 | 870.00 | 2025-09-03 10:15:00 | 957.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-11 14:30:00 | 870.00 | 2025-09-03 10:15:00 | 957.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-18 09:15:00 | 871.20 | 2025-09-03 10:15:00 | 958.32 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-18 09:45:00 | 889.75 | 2025-10-13 10:15:00 | 820.60 | STOP_HIT | 1.00 | -7.77% |
| BUY | retest1 | 2025-12-02 14:45:00 | 988.70 | 2025-12-05 11:15:00 | 936.70 | STOP_HIT | 1.00 | -5.26% |
| BUY | retest1 | 2025-12-03 09:30:00 | 992.80 | 2025-12-05 11:15:00 | 936.70 | STOP_HIT | 1.00 | -5.65% |
| BUY | retest1 | 2025-12-04 11:15:00 | 990.90 | 2025-12-05 11:15:00 | 936.70 | STOP_HIT | 1.00 | -5.47% |
| BUY | retest2 | 2025-12-05 14:45:00 | 951.10 | 2025-12-08 12:15:00 | 932.80 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-12-08 09:30:00 | 965.20 | 2025-12-08 12:15:00 | 932.80 | STOP_HIT | 1.00 | -3.36% |
| BUY | retest2 | 2025-12-09 10:15:00 | 950.50 | 2025-12-29 10:15:00 | 929.20 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2025-12-23 14:15:00 | 950.30 | 2025-12-29 10:15:00 | 929.20 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2026-01-13 15:15:00 | 975.00 | 2026-01-21 09:15:00 | 943.90 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest2 | 2026-01-14 10:30:00 | 972.75 | 2026-01-21 09:15:00 | 943.90 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2026-01-16 09:30:00 | 973.20 | 2026-01-21 09:15:00 | 943.90 | STOP_HIT | 1.00 | -3.01% |
| BUY | retest2 | 2026-01-16 14:45:00 | 972.40 | 2026-01-21 09:15:00 | 943.90 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2026-01-21 15:15:00 | 953.00 | 2026-01-22 15:15:00 | 930.00 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2026-01-27 14:45:00 | 951.10 | 2026-01-29 09:15:00 | 922.45 | STOP_HIT | 1.00 | -3.01% |
| BUY | retest2 | 2026-01-28 10:30:00 | 952.10 | 2026-01-29 09:15:00 | 922.45 | STOP_HIT | 1.00 | -3.11% |
| BUY | retest2 | 2026-01-28 11:00:00 | 950.80 | 2026-01-29 09:15:00 | 922.45 | STOP_HIT | 1.00 | -2.98% |
| BUY | retest2 | 2026-02-02 14:00:00 | 972.15 | 2026-02-23 13:15:00 | 1069.37 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-02 14:45:00 | 966.75 | 2026-02-23 13:15:00 | 1063.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-03 11:30:00 | 971.80 | 2026-02-23 13:15:00 | 1068.98 | TARGET_HIT | 1.00 | 10.00% |
