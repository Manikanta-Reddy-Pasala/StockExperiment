# Fertilisers and Chemicals Travancore Ltd. (FACT)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 902.80
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
| ALERT2_SKIP | 0 |
| ALERT3 | 10 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 6 |
| PARTIAL | 0 |
| TARGET_HIT | 1 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / Stop hits / Partials:** 1 / 5 / 0
- **Avg / median % per leg:** 0.14% / -1.33%
- **Sum % (uncompounded):** 0.86%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 1 | 16.7% | 1 | 5 | 0 | 0.14% | 0.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 1 | 16.7% | 1 | 5 | 0 | 0.14% | 0.9% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 6 | 1 | 16.7% | 1 | 5 | 0 | 0.14% | 0.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 11:15:00 | 830.00 | 745.50 | 745.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 14:15:00 | 834.00 | 747.97 | 746.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 14:15:00 | 967.40 | 971.67 | 906.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 15:00:00 | 967.40 | 971.67 | 906.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 916.70 | 957.71 | 918.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 916.70 | 957.71 | 918.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 917.85 | 957.31 | 918.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 09:15:00 | 946.55 | 952.37 | 918.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 14:00:00 | 923.95 | 952.23 | 934.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 09:15:00 | 925.40 | 950.32 | 934.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 09:15:00 | 911.70 | 949.94 | 934.13 | SL hit (close<static) qty=1.00 sl=915.45 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-11 09:15:00 | 911.70 | 949.94 | 934.13 | SL hit (close<static) qty=1.00 sl=915.45 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-11 09:15:00 | 911.70 | 949.94 | 934.13 | SL hit (close<static) qty=1.00 sl=915.45 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 09:15:00 | 930.90 | 947.77 | 933.50 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 935.80 | 947.53 | 933.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 11:00:00 | 935.80 | 947.53 | 933.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 11:15:00 | 934.75 | 947.41 | 933.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 11:30:00 | 930.05 | 947.41 | 933.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 12:15:00 | 929.95 | 947.23 | 933.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 13:00:00 | 929.95 | 947.23 | 933.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 13:15:00 | 929.00 | 947.05 | 933.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 14:00:00 | 929.00 | 947.05 | 933.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 985.95 | 947.32 | 934.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:30:00 | 991.00 | 947.32 | 934.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Target hit | 2025-08-21 09:15:00 | 1023.99 | 956.64 | 940.92 | Target hit (10%) qty=1.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 944.90 | 967.39 | 948.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 09:30:00 | 943.60 | 967.39 | 948.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 943.50 | 967.15 | 948.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 10:30:00 | 942.00 | 967.15 | 948.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 948.80 | 961.49 | 947.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 15:15:00 | 954.00 | 961.49 | 947.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-02 09:30:00 | 955.40 | 961.57 | 947.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-26 11:15:00 | 942.05 | 978.96 | 966.15 | SL hit (close<static) qty=1.00 sl=945.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-26 11:15:00 | 942.05 | 978.96 | 966.15 | SL hit (close<static) qty=1.00 sl=945.00 alert=retest2 |
| CROSSOVER_SKIP | 2025-10-06 14:15:00 | 898.00 | 955.29 | 955.54 | min_gap filter: gap=0.028% < 0.030% |
| TREND_RESET | 2025-10-06 14:15:00 | 898.00 | 955.29 | 955.54 | EMA inversion without crossover edge (EMA200=955.29 EMA400=955.54) — end cycle |
| CROSSOVER_SKIP | 2026-04-16 15:15:00 | 870.00 | 807.96 | 807.88 | min_gap filter: gap=0.009% < 0.030% |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-15 09:15:00 | 946.55 | 2025-08-11 09:15:00 | 911.70 | STOP_HIT | 1.00 | -3.68% |
| BUY | retest2 | 2025-08-07 14:00:00 | 923.95 | 2025-08-11 09:15:00 | 911.70 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-08-11 09:15:00 | 925.40 | 2025-08-11 09:15:00 | 911.70 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-08-12 09:15:00 | 930.90 | 2025-08-21 09:15:00 | 1023.99 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-01 15:15:00 | 954.00 | 2025-09-26 11:15:00 | 942.05 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-09-02 09:30:00 | 955.40 | 2025-09-26 11:15:00 | 942.05 | STOP_HIT | 1.00 | -1.40% |
