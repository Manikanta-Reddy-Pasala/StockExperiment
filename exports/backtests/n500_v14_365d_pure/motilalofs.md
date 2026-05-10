# Motilal Oswal Financial Services Ltd. (MOTILALOFS)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 882.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 18 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 14 |
| PARTIAL | 0 |
| TARGET_HIT | 2 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 12
- **Target hits / Stop hits / Partials:** 2 / 12 / 0
- **Avg / median % per leg:** -0.51% / -1.64%
- **Sum % (uncompounded):** -7.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 2 | 14.3% | 2 | 12 | 0 | -0.51% | -7.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 14 | 2 | 14.3% | 2 | 12 | 0 | -0.51% | -7.1% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 14 | 2 | 14.3% | 2 | 12 | 0 | -0.51% | -7.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 14:15:00 | 740.00 | 671.78 | 671.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 11:15:00 | 748.00 | 674.50 | 672.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 12:15:00 | 894.70 | 895.70 | 849.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-28 12:45:00 | 896.75 | 895.70 | 849.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 14:15:00 | 891.90 | 919.18 | 889.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 15:00:00 | 891.90 | 919.18 | 889.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 15:15:00 | 889.00 | 918.88 | 889.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 09:30:00 | 883.90 | 918.42 | 889.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 881.40 | 918.05 | 889.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 12:15:00 | 883.35 | 917.68 | 889.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-29 13:15:00 | 868.85 | 916.79 | 888.99 | SL hit (close<static) qty=1.00 sl=872.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 09:45:00 | 883.70 | 915.24 | 888.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 10:15:00 | 883.95 | 915.24 | 888.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 11:15:00 | 883.90 | 914.91 | 888.59 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 890.25 | 914.09 | 888.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:30:00 | 888.90 | 914.09 | 888.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 893.70 | 913.89 | 888.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-01 15:00:00 | 893.70 | 913.89 | 888.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 873.75 | 913.30 | 888.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:00:00 | 873.75 | 913.30 | 888.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 873.65 | 912.91 | 888.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 11:00:00 | 873.65 | 912.91 | 888.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 877.60 | 908.46 | 887.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:45:00 | 879.00 | 908.46 | 887.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-04 12:15:00 | 871.65 | 907.74 | 887.67 | SL hit (close<static) qty=1.00 sl=872.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-04 12:15:00 | 871.65 | 907.74 | 887.67 | SL hit (close<static) qty=1.00 sl=872.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-04 12:15:00 | 871.65 | 907.74 | 887.67 | SL hit (close<static) qty=1.00 sl=872.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 11:15:00 | 889.60 | 903.86 | 886.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 12:15:00 | 894.30 | 903.86 | 886.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 10:15:00 | 881.85 | 903.24 | 887.12 | SL hit (close<static) qty=1.00 sl=884.75 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 895.55 | 902.29 | 887.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 15:15:00 | 880.10 | 919.73 | 904.41 | SL hit (close<static) qty=1.00 sl=884.75 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 13:00:00 | 894.00 | 918.41 | 904.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 13:45:00 | 891.95 | 918.16 | 903.99 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 904.50 | 917.59 | 903.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 10:45:00 | 902.60 | 917.59 | 903.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 11:15:00 | 904.00 | 917.46 | 903.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 11:30:00 | 906.10 | 917.46 | 903.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 12:15:00 | 902.10 | 917.30 | 903.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 12:30:00 | 901.30 | 917.30 | 903.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 13:15:00 | 898.70 | 917.12 | 903.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 13:30:00 | 898.35 | 917.12 | 903.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Target hit | 2025-10-09 13:15:00 | 983.40 | 921.88 | 908.17 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-10-09 13:15:00 | 981.15 | 921.88 | 908.17 | Target hit (10%) qty=1.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 946.70 | 983.58 | 950.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 946.70 | 983.58 | 950.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 964.00 | 983.39 | 950.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 14:15:00 | 975.60 | 983.04 | 951.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 09:30:00 | 971.30 | 984.71 | 955.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 11:45:00 | 972.45 | 987.41 | 964.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 15:00:00 | 970.00 | 986.97 | 964.73 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 966.60 | 986.77 | 964.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 09:15:00 | 972.20 | 986.77 | 964.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 11:15:00 | 956.20 | 986.09 | 964.72 | SL hit (close<static) qty=1.00 sl=963.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 09:15:00 | 937.65 | 982.25 | 964.00 | SL hit (close<static) qty=1.00 sl=945.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 09:15:00 | 937.65 | 982.25 | 964.00 | SL hit (close<static) qty=1.00 sl=945.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 09:15:00 | 937.65 | 982.25 | 964.00 | SL hit (close<static) qty=1.00 sl=945.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 09:15:00 | 937.65 | 982.25 | 964.00 | SL hit (close<static) qty=1.00 sl=945.55 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 09:30:00 | 981.45 | 970.74 | 960.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 14:15:00 | 959.05 | 970.79 | 960.93 | SL hit (close<static) qty=1.00 sl=963.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-12-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 14:15:00 | 844.60 | 953.12 | 953.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 826.00 | 950.82 | 952.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 09:15:00 | 870.35 | 863.51 | 890.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-16 09:30:00 | 870.25 | 863.51 | 890.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 745.00 | 700.56 | 740.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 10:00:00 | 745.00 | 700.56 | 740.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 749.30 | 701.05 | 740.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 11:00:00 | 749.30 | 701.05 | 740.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2026-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 15:15:00 | 807.00 | 762.39 | 762.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 822.55 | 762.99 | 762.51 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-08-29 12:15:00 | 883.35 | 2025-08-29 13:15:00 | 868.85 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-09-01 09:45:00 | 883.70 | 2025-09-04 12:15:00 | 871.65 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-09-01 10:15:00 | 883.95 | 2025-09-04 12:15:00 | 871.65 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-09-01 11:15:00 | 883.90 | 2025-09-04 12:15:00 | 871.65 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-09-08 12:15:00 | 894.30 | 2025-09-09 10:15:00 | 881.85 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-09-10 09:15:00 | 895.55 | 2025-09-30 15:15:00 | 880.10 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-10-01 13:00:00 | 894.00 | 2025-10-09 13:15:00 | 983.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-01 13:45:00 | 891.95 | 2025-10-09 13:15:00 | 981.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-31 14:15:00 | 975.60 | 2025-11-20 11:15:00 | 956.20 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2025-11-07 09:30:00 | 971.30 | 2025-11-24 09:15:00 | 937.65 | STOP_HIT | 1.00 | -3.46% |
| BUY | retest2 | 2025-11-19 11:45:00 | 972.45 | 2025-11-24 09:15:00 | 937.65 | STOP_HIT | 1.00 | -3.58% |
| BUY | retest2 | 2025-11-19 15:00:00 | 970.00 | 2025-11-24 09:15:00 | 937.65 | STOP_HIT | 1.00 | -3.34% |
| BUY | retest2 | 2025-11-20 09:15:00 | 972.20 | 2025-11-24 09:15:00 | 937.65 | STOP_HIT | 1.00 | -3.55% |
| BUY | retest2 | 2025-12-01 09:30:00 | 981.45 | 2025-12-01 14:15:00 | 959.05 | STOP_HIT | 1.00 | -2.28% |
