# BAJFINANCE (BAJFINANCE)

## Backtest Summary

- **Window:** 2024-09-18 09:15:00 → 2026-05-08 15:15:00 (2825 bars)
- **Last close:** 954.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 3 |
| ALERT3 | 25 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 25 |
| PARTIAL | 0 |
| TARGET_HIT | 4 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 21
- **Target hits / Stop hits / Partials:** 4 / 21 / 0
- **Avg / median % per leg:** 0.14% / -1.37%
- **Sum % (uncompounded):** 3.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 4 | 40.0% | 4 | 6 | 0 | 3.15% | 31.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 10 | 4 | 40.0% | 4 | 6 | 0 | 3.15% | 31.5% |
| SELL (all) | 15 | 0 | 0.0% | 0 | 15 | 0 | -1.87% | -28.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 15 | 0 | 0.0% | 0 | 15 | 0 | -1.87% | -28.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 25 | 4 | 16.0% | 4 | 21 | 0 | 0.14% | 3.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-01-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-13 09:15:00 | 721.29 | 707.18 | 707.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-14 09:15:00 | 730.72 | 708.03 | 707.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 10:15:00 | 849.50 | 859.86 | 828.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-03 11:00:00 | 849.50 | 859.86 | 828.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 861.35 | 889.77 | 859.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:30:00 | 867.40 | 889.77 | 859.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 10:15:00 | 860.35 | 889.48 | 859.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 10:30:00 | 860.90 | 889.48 | 859.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 11:15:00 | 858.50 | 889.17 | 859.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 11:45:00 | 859.20 | 889.17 | 859.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 12:15:00 | 861.55 | 888.89 | 859.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-30 13:30:00 | 864.10 | 888.66 | 859.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 09:15:00 | 864.30 | 888.14 | 859.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 09:45:00 | 867.85 | 887.93 | 859.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 15:00:00 | 863.55 | 886.58 | 864.12 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 15:15:00 | 866.20 | 886.38 | 864.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 09:15:00 | 894.20 | 886.38 | 864.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-09 09:15:00 | 950.51 | 907.78 | 889.72 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 15:15:00 | 876.95 | 911.55 | 911.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 872.05 | 911.16 | 911.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 915.20 | 896.50 | 903.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 915.20 | 896.50 | 903.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 915.20 | 896.50 | 903.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:30:00 | 915.00 | 896.50 | 903.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 912.55 | 896.66 | 903.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 13:45:00 | 908.80 | 897.11 | 903.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 09:15:00 | 936.50 | 894.07 | 899.59 | SL hit (close>static) qty=1.00 sl=918.75 alert=retest2 |

### Cycle 3 — BUY (started 2025-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 14:15:00 | 948.95 | 904.73 | 904.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 961.00 | 905.71 | 905.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 09:15:00 | 1007.70 | 1040.32 | 1007.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 09:15:00 | 1007.70 | 1040.32 | 1007.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 1007.70 | 1040.32 | 1007.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:30:00 | 1011.90 | 1040.32 | 1007.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 1002.70 | 1039.94 | 1007.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:00:00 | 1002.70 | 1039.94 | 1007.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 1001.10 | 1039.56 | 1007.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 15:15:00 | 1009.40 | 1038.50 | 1007.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 15:15:00 | 1006.70 | 1035.04 | 1007.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 10:45:00 | 1008.40 | 1034.21 | 1007.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 10:30:00 | 1007.20 | 1027.77 | 1009.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 13:15:00 | 994.80 | 1026.90 | 1009.08 | SL hit (close<static) qty=1.00 sl=997.00 alert=retest2 |

### Cycle 4 — SELL (started 2026-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 15:15:00 | 975.00 | 1008.84 | 1008.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 09:15:00 | 968.45 | 1003.44 | 1006.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 11:15:00 | 955.95 | 954.92 | 974.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-03 12:00:00 | 955.95 | 954.92 | 974.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 969.95 | 955.52 | 974.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:00:00 | 969.95 | 955.52 | 974.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 972.00 | 956.66 | 973.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:45:00 | 977.90 | 956.66 | 973.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 972.05 | 956.81 | 973.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 10:00:00 | 964.50 | 959.49 | 973.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 11:00:00 | 964.70 | 959.54 | 973.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 12:00:00 | 965.30 | 959.60 | 973.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 15:00:00 | 965.50 | 959.84 | 973.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 970.50 | 960.03 | 973.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:45:00 | 972.00 | 960.03 | 973.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 977.95 | 960.63 | 973.41 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 977.95 | 960.63 | 973.41 | SL hit (close>static) qty=1.00 sl=975.90 alert=retest2 |

### Cycle 5 — BUY (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 11:15:00 | 1037.60 | 983.25 | 983.13 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2026-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 15:15:00 | 951.70 | 984.31 | 984.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 925.45 | 983.72 | 984.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 906.60 | 885.10 | 920.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 10:15:00 | 921.70 | 885.46 | 920.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 921.70 | 885.46 | 920.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 11:00:00 | 921.70 | 885.46 | 920.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 11:15:00 | 918.10 | 885.79 | 920.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:15:00 | 908.80 | 886.96 | 920.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 898.85 | 890.44 | 920.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 13:15:00 | 912.50 | 892.21 | 919.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 15:15:00 | 912.50 | 892.64 | 919.31 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 931.20 | 893.22 | 919.34 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-16 09:15:00 | 931.20 | 893.22 | 919.34 | SL hit (close>static) qty=1.00 sl=926.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-04-30 13:30:00 | 864.10 | 2025-06-09 09:15:00 | 950.51 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-02 09:15:00 | 864.30 | 2025-06-09 09:15:00 | 950.73 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-02 09:45:00 | 867.85 | 2025-06-09 09:15:00 | 954.64 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-09 15:00:00 | 863.55 | 2025-06-09 09:15:00 | 949.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-12 09:15:00 | 894.20 | 2025-08-06 15:15:00 | 876.95 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2025-08-18 13:45:00 | 908.80 | 2025-09-04 09:15:00 | 936.50 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2025-11-11 15:15:00 | 1009.40 | 2025-11-24 13:15:00 | 994.80 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-11-13 15:15:00 | 1006.70 | 2025-11-24 13:15:00 | 994.80 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-11-14 10:45:00 | 1008.40 | 2025-11-24 13:15:00 | 994.80 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-11-24 10:30:00 | 1007.20 | 2025-11-24 13:15:00 | 994.80 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-12-24 09:15:00 | 1022.90 | 2025-12-24 15:15:00 | 1009.40 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2026-02-10 10:00:00 | 964.50 | 2026-02-12 09:15:00 | 977.95 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2026-02-10 11:00:00 | 964.70 | 2026-02-12 09:15:00 | 977.95 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2026-02-10 12:00:00 | 965.30 | 2026-02-12 09:15:00 | 977.95 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2026-02-10 15:00:00 | 965.50 | 2026-02-12 09:15:00 | 977.95 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2026-04-09 09:15:00 | 908.80 | 2026-04-16 09:15:00 | 931.20 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2026-04-13 09:15:00 | 898.85 | 2026-04-16 09:15:00 | 931.20 | STOP_HIT | 1.00 | -3.60% |
| SELL | retest2 | 2026-04-15 13:15:00 | 912.50 | 2026-04-16 09:15:00 | 931.20 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2026-04-15 15:15:00 | 912.50 | 2026-04-16 09:15:00 | 931.20 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2026-04-16 11:15:00 | 920.50 | 2026-04-21 11:15:00 | 939.75 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2026-04-23 09:30:00 | 920.05 | 2026-04-24 15:15:00 | 923.00 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2026-04-23 10:00:00 | 919.85 | 2026-04-27 13:15:00 | 926.90 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2026-04-23 14:00:00 | 921.70 | 2026-04-29 12:15:00 | 936.40 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2026-04-23 15:15:00 | 916.50 | 2026-04-29 12:15:00 | 936.40 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2026-04-27 10:00:00 | 913.95 | 2026-04-29 12:15:00 | 936.40 | STOP_HIT | 1.00 | -2.46% |
