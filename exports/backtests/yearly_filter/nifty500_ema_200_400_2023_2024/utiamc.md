# UTI Asset Management Company Ltd. (UTIAMC)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 973.15
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 0 |
| ALERT3 | 29 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 17 |
| PARTIAL | 5 |
| TARGET_HIT | 4 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 11
- **Target hits / Stop hits / Partials:** 4 / 13 / 5
- **Avg / median % per leg:** 2.29% / 1.64%
- **Sum % (uncompounded):** 50.44%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 3 | 42.9% | 3 | 4 | 0 | 2.99% | 20.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 3 | 42.9% | 3 | 4 | 0 | 2.99% | 20.9% |
| SELL (all) | 15 | 8 | 53.3% | 1 | 9 | 5 | 1.97% | 29.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 15 | 8 | 53.3% | 1 | 9 | 5 | 1.97% | 29.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 22 | 11 | 50.0% | 4 | 13 | 5 | 2.29% | 50.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-20 15:15:00 | 817.00 | 872.97 | 873.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-27 14:15:00 | 812.15 | 862.17 | 867.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-03 10:15:00 | 857.40 | 856.50 | 863.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-03 11:00:00 | 857.40 | 856.50 | 863.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 12:15:00 | 862.00 | 856.54 | 863.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-03 13:00:00 | 862.00 | 856.54 | 863.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 13:15:00 | 864.55 | 856.62 | 863.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-03 14:00:00 | 864.55 | 856.62 | 863.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 14:15:00 | 866.00 | 856.72 | 863.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-03 15:15:00 | 862.20 | 856.72 | 863.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-04 09:15:00 | 870.60 | 856.91 | 863.64 | SL hit (close>static) qty=1.00 sl=869.25 alert=retest2 |

### Cycle 2 — BUY (started 2024-04-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-09 12:15:00 | 933.80 | 869.55 | 869.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 12:15:00 | 943.90 | 885.58 | 878.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-07 10:15:00 | 914.60 | 916.63 | 898.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-07 10:45:00 | 916.15 | 916.63 | 898.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 14:15:00 | 898.30 | 916.39 | 900.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 15:00:00 | 898.30 | 916.39 | 900.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 15:15:00 | 897.00 | 916.19 | 900.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-10 09:15:00 | 904.00 | 916.19 | 900.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-13 11:15:00 | 902.35 | 915.45 | 900.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-14 09:15:00 | 906.75 | 914.84 | 900.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-12 09:15:00 | 992.59 | 932.01 | 917.57 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-01-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 12:15:00 | 1221.25 | 1273.44 | 1273.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 14:15:00 | 1214.55 | 1272.32 | 1273.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 14:15:00 | 987.50 | 984.19 | 1053.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-19 14:45:00 | 991.35 | 984.19 | 1053.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 1063.10 | 987.03 | 1049.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 15:00:00 | 1063.10 | 987.03 | 1049.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 15:15:00 | 1056.25 | 987.72 | 1049.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 09:15:00 | 1059.30 | 987.72 | 1049.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 10:15:00 | 1058.95 | 989.13 | 1049.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 10:30:00 | 1063.85 | 989.13 | 1049.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 1049.55 | 996.78 | 1050.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-25 14:00:00 | 1049.55 | 996.78 | 1050.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 14:15:00 | 1060.65 | 997.41 | 1050.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-25 15:00:00 | 1060.65 | 997.41 | 1050.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 1065.00 | 998.08 | 1051.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:15:00 | 1055.90 | 998.08 | 1051.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 1045.85 | 999.04 | 1050.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-26 10:30:00 | 1048.60 | 999.04 | 1050.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 1038.15 | 1001.32 | 1050.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 09:15:00 | 1013.15 | 1021.30 | 1052.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 962.49 | 1021.13 | 1052.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-07 15:15:00 | 1023.30 | 1020.79 | 1051.11 | SL hit (close>ema200) qty=0.50 sl=1020.79 alert=retest2 |

### Cycle 4 — BUY (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 10:15:00 | 1175.40 | 1055.16 | 1054.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 12:15:00 | 1189.00 | 1057.64 | 1056.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 11:15:00 | 1341.40 | 1350.25 | 1275.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-25 11:45:00 | 1342.10 | 1350.25 | 1275.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 1312.40 | 1348.57 | 1312.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:45:00 | 1312.90 | 1348.57 | 1312.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 12:15:00 | 1304.30 | 1348.13 | 1312.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 13:00:00 | 1304.30 | 1348.13 | 1312.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 1294.30 | 1340.25 | 1311.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-01 15:00:00 | 1294.30 | 1340.25 | 1311.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 1332.10 | 1353.39 | 1332.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:30:00 | 1331.40 | 1353.39 | 1332.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 12:15:00 | 1330.00 | 1353.16 | 1332.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 13:00:00 | 1330.00 | 1353.16 | 1332.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 1314.90 | 1352.78 | 1332.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 14:00:00 | 1314.90 | 1352.78 | 1332.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 1322.00 | 1344.74 | 1330.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 13:45:00 | 1309.80 | 1344.74 | 1330.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 1302.00 | 1343.98 | 1329.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:15:00 | 1317.60 | 1343.98 | 1329.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 1326.70 | 1338.62 | 1328.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 09:15:00 | 1344.60 | 1330.59 | 1325.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 1300.00 | 1343.83 | 1333.30 | SL hit (close<static) qty=1.00 sl=1321.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 09:15:00 | 1243.50 | 1325.78 | 1325.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 10:15:00 | 1236.00 | 1319.56 | 1322.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 14:15:00 | 1140.50 | 1139.29 | 1176.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:45:00 | 1141.90 | 1139.29 | 1176.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 1106.40 | 1058.80 | 1099.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 09:15:00 | 1083.45 | 1061.09 | 1099.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 10:15:00 | 1091.70 | 1061.43 | 1099.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 13:45:00 | 1091.55 | 1062.52 | 1099.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 09:15:00 | 1037.12 | 1065.28 | 1094.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 09:15:00 | 1036.97 | 1065.28 | 1094.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-20 11:15:00 | 1065.70 | 1065.28 | 1094.03 | SL hit (close>ema200) qty=0.50 sl=1065.28 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-03 15:15:00 | 862.20 | 2024-04-04 09:15:00 | 870.60 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-05-10 09:15:00 | 904.00 | 2024-06-12 09:15:00 | 992.59 | TARGET_HIT | 1.00 | 9.80% |
| BUY | retest2 | 2024-05-13 11:15:00 | 902.35 | 2024-06-12 11:15:00 | 994.40 | TARGET_HIT | 1.00 | 10.20% |
| BUY | retest2 | 2024-05-14 09:15:00 | 906.75 | 2024-06-13 09:15:00 | 997.43 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-07 09:15:00 | 1013.15 | 2025-04-07 09:15:00 | 962.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-07 09:15:00 | 1013.15 | 2025-04-07 15:15:00 | 1023.30 | STOP_HIT | 0.50 | -1.00% |
| SELL | retest2 | 2025-04-08 09:45:00 | 1030.80 | 2025-04-21 09:15:00 | 1054.00 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2025-04-08 10:30:00 | 1031.15 | 2025-04-21 09:15:00 | 1054.00 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2025-04-08 11:00:00 | 1031.15 | 2025-04-21 09:15:00 | 1054.00 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2025-04-30 13:15:00 | 1042.60 | 2025-05-06 14:15:00 | 990.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-30 13:15:00 | 1042.60 | 2025-05-12 14:15:00 | 1045.00 | STOP_HIT | 0.50 | -0.23% |
| SELL | retest2 | 2025-05-13 09:15:00 | 1042.60 | 2025-05-13 10:15:00 | 1056.10 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-10-13 09:15:00 | 1344.60 | 2025-10-20 09:15:00 | 1300.00 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest2 | 2025-10-20 12:30:00 | 1334.50 | 2025-10-23 09:15:00 | 1308.00 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2025-10-20 13:00:00 | 1334.90 | 2025-10-23 09:15:00 | 1308.00 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-10-24 14:30:00 | 1341.30 | 2025-10-27 11:15:00 | 1318.00 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2026-02-12 09:15:00 | 1083.45 | 2026-02-20 09:15:00 | 1037.12 | PARTIAL | 0.50 | 4.28% |
| SELL | retest2 | 2026-02-12 10:15:00 | 1091.70 | 2026-02-20 09:15:00 | 1036.97 | PARTIAL | 0.50 | 5.01% |
| SELL | retest2 | 2026-02-12 09:15:00 | 1083.45 | 2026-02-20 11:15:00 | 1065.70 | STOP_HIT | 0.50 | 1.64% |
| SELL | retest2 | 2026-02-12 10:15:00 | 1091.70 | 2026-02-20 11:15:00 | 1065.70 | STOP_HIT | 0.50 | 2.38% |
| SELL | retest2 | 2026-02-12 13:45:00 | 1091.55 | 2026-02-26 12:15:00 | 1029.28 | PARTIAL | 0.50 | 5.70% |
| SELL | retest2 | 2026-02-12 13:45:00 | 1091.55 | 2026-03-02 09:15:00 | 975.11 | TARGET_HIT | 0.50 | 10.67% |
