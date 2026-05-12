# CreditAccess Grameen Ltd. (CREDITACC)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 1493.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 1 |
| ALERT3 | 36 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 45 |
| PARTIAL | 20 |
| TARGET_HIT | 17 |
| STOP_HIT | 29 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 66 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 40 / 26
- **Target hits / Stop hits / Partials:** 17 / 29 / 20
- **Avg / median % per leg:** 2.75% / 5.00%
- **Sum % (uncompounded):** 181.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 9 | 64.3% | 8 | 5 | 1 | 4.00% | 56.0% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| BUY @ 3rd Alert (retest2) | 12 | 7 | 58.3% | 7 | 5 | 0 | 3.42% | 41.0% |
| SELL (all) | 52 | 31 | 59.6% | 9 | 24 | 19 | 2.41% | 125.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 52 | 31 | 59.6% | 9 | 24 | 19 | 2.41% | 125.2% |
| retest1 (combined) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| retest2 (combined) | 64 | 38 | 59.4% | 16 | 29 | 19 | 2.60% | 166.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-19 09:15:00 | 1225.05 | 1178.60 | 1110.64 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-20 09:15:00 | 1286.30 | 1184.50 | 1116.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2023-06-23 11:15:00 | 1347.56 | 1210.00 | 1137.13 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 2 — SELL (started 2024-02-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-15 14:15:00 | 1557.00 | 1617.82 | 1617.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-19 09:15:00 | 1549.00 | 1613.11 | 1615.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 09:15:00 | 1449.80 | 1436.21 | 1490.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-01 09:30:00 | 1448.70 | 1436.21 | 1490.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 09:15:00 | 1478.65 | 1432.01 | 1482.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-19 09:15:00 | 1411.50 | 1446.45 | 1477.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-23 10:15:00 | 1486.45 | 1447.85 | 1476.23 | SL hit (close>static) qty=1.00 sl=1484.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-02-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-03 15:15:00 | 1008.00 | 962.58 | 962.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 09:15:00 | 1019.40 | 963.15 | 962.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-11 15:15:00 | 988.00 | 990.41 | 977.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-12 09:15:00 | 981.55 | 990.41 | 977.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 09:15:00 | 966.65 | 990.18 | 977.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-12 10:00:00 | 966.65 | 990.18 | 977.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 10:15:00 | 997.65 | 990.25 | 977.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-12 11:45:00 | 1014.75 | 990.40 | 978.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-13 09:45:00 | 1013.55 | 990.95 | 978.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-13 11:00:00 | 1019.10 | 991.23 | 978.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-17 09:15:00 | 945.10 | 991.60 | 979.87 | SL hit (close<static) qty=1.00 sl=961.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 10:15:00 | 862.30 | 970.40 | 970.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 12:15:00 | 851.30 | 968.17 | 969.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 09:15:00 | 973.40 | 959.26 | 964.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 09:15:00 | 973.40 | 959.26 | 964.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 973.40 | 959.26 | 964.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 10:00:00 | 973.40 | 959.26 | 964.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 10:15:00 | 952.50 | 959.19 | 964.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 09:15:00 | 909.50 | 959.63 | 964.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 11:45:00 | 929.45 | 958.75 | 964.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 12:15:00 | 929.95 | 958.75 | 964.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 09:30:00 | 918.40 | 956.77 | 963.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-04 09:15:00 | 882.98 | 953.16 | 961.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-04 09:15:00 | 883.45 | 953.16 | 961.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-06 09:15:00 | 949.65 | 948.41 | 958.14 | SL hit (close>ema200) qty=0.50 sl=948.41 alert=retest2 |

### Cycle 5 — BUY (started 2025-04-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-04 15:15:00 | 998.95 | 960.10 | 959.95 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-04-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 11:15:00 | 949.65 | 959.74 | 959.77 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 11:15:00 | 988.65 | 960.02 | 959.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 12:15:00 | 994.95 | 960.37 | 960.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 09:15:00 | 1103.10 | 1111.14 | 1062.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 09:30:00 | 1100.00 | 1111.14 | 1062.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 1119.60 | 1158.76 | 1120.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:00:00 | 1119.60 | 1158.76 | 1120.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 1122.10 | 1158.40 | 1120.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:30:00 | 1116.90 | 1158.40 | 1120.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 1120.60 | 1158.02 | 1120.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:45:00 | 1120.00 | 1158.02 | 1120.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 1122.50 | 1157.67 | 1120.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 09:15:00 | 1133.60 | 1156.92 | 1120.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 12:15:00 | 1108.20 | 1155.54 | 1120.18 | SL hit (close<static) qty=1.00 sl=1116.20 alert=retest2 |

### Cycle 8 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 1255.30 | 1350.52 | 1350.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 1252.70 | 1349.55 | 1350.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 09:15:00 | 1313.10 | 1310.90 | 1326.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-24 10:00:00 | 1313.10 | 1310.90 | 1326.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 1317.50 | 1297.83 | 1315.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:30:00 | 1324.00 | 1297.83 | 1315.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 1305.10 | 1297.91 | 1315.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 09:15:00 | 1303.50 | 1298.76 | 1315.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 10:00:00 | 1301.30 | 1298.79 | 1315.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 11:15:00 | 1322.30 | 1299.19 | 1315.82 | SL hit (close>static) qty=1.00 sl=1318.20 alert=retest2 |

### Cycle 9 — BUY (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 10:15:00 | 1415.80 | 1247.29 | 1247.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 11:15:00 | 1446.00 | 1249.26 | 1248.04 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-06-19 09:15:00 | 1225.05 | 2023-06-20 09:15:00 | 1286.30 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2023-06-19 09:15:00 | 1225.05 | 2023-06-23 11:15:00 | 1347.56 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2023-10-11 12:15:00 | 1358.00 | 2023-10-23 09:15:00 | 1493.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-10-11 14:00:00 | 1366.00 | 2023-10-23 09:15:00 | 1502.60 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-04-19 09:15:00 | 1411.50 | 2024-04-23 10:15:00 | 1486.45 | STOP_HIT | 1.00 | -5.31% |
| SELL | retest2 | 2024-05-07 15:15:00 | 1414.00 | 2024-05-28 15:15:00 | 1343.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-09 09:30:00 | 1411.90 | 2024-05-28 15:15:00 | 1341.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-09 15:00:00 | 1397.20 | 2024-05-29 09:15:00 | 1327.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-07 15:15:00 | 1414.00 | 2024-06-04 10:15:00 | 1272.60 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-09 09:30:00 | 1411.90 | 2024-06-04 10:15:00 | 1270.71 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-09 15:00:00 | 1397.20 | 2024-06-04 10:15:00 | 1257.48 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-24 12:15:00 | 1409.05 | 2024-06-27 12:15:00 | 1338.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-24 12:15:00 | 1409.05 | 2024-07-10 09:15:00 | 1268.14 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-12 11:45:00 | 1014.75 | 2025-02-17 09:15:00 | 945.10 | STOP_HIT | 1.00 | -6.86% |
| BUY | retest2 | 2025-02-13 09:45:00 | 1013.55 | 2025-02-17 09:15:00 | 945.10 | STOP_HIT | 1.00 | -6.75% |
| BUY | retest2 | 2025-02-13 11:00:00 | 1019.10 | 2025-02-17 09:15:00 | 945.10 | STOP_HIT | 1.00 | -7.26% |
| BUY | retest2 | 2025-02-17 15:00:00 | 1016.75 | 2025-02-18 15:15:00 | 956.90 | STOP_HIT | 1.00 | -5.89% |
| SELL | retest2 | 2025-02-28 09:15:00 | 909.50 | 2025-03-04 09:15:00 | 882.98 | PARTIAL | 0.50 | 2.92% |
| SELL | retest2 | 2025-02-28 11:45:00 | 929.45 | 2025-03-04 09:15:00 | 883.45 | PARTIAL | 0.50 | 4.95% |
| SELL | retest2 | 2025-02-28 09:15:00 | 909.50 | 2025-03-06 09:15:00 | 949.65 | STOP_HIT | 0.50 | -4.41% |
| SELL | retest2 | 2025-02-28 11:45:00 | 929.45 | 2025-03-06 09:15:00 | 949.65 | STOP_HIT | 0.50 | -2.17% |
| SELL | retest2 | 2025-02-28 12:15:00 | 929.95 | 2025-03-06 10:15:00 | 998.40 | STOP_HIT | 1.00 | -7.36% |
| SELL | retest2 | 2025-03-03 09:30:00 | 918.40 | 2025-03-06 10:15:00 | 998.40 | STOP_HIT | 1.00 | -8.71% |
| SELL | retest2 | 2025-03-06 15:00:00 | 982.75 | 2025-03-10 09:15:00 | 933.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-07 09:45:00 | 976.45 | 2025-03-10 15:15:00 | 927.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-07 12:00:00 | 974.30 | 2025-03-11 09:15:00 | 925.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-06 15:00:00 | 982.75 | 2025-03-19 09:15:00 | 953.25 | STOP_HIT | 0.50 | 3.00% |
| SELL | retest2 | 2025-03-07 09:45:00 | 976.45 | 2025-03-19 09:15:00 | 953.25 | STOP_HIT | 0.50 | 2.38% |
| SELL | retest2 | 2025-03-07 12:00:00 | 974.30 | 2025-03-19 09:15:00 | 953.25 | STOP_HIT | 0.50 | 2.16% |
| SELL | retest2 | 2025-03-25 11:00:00 | 974.95 | 2025-03-26 10:15:00 | 926.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-25 11:00:00 | 974.95 | 2025-03-27 14:15:00 | 980.55 | STOP_HIT | 0.50 | -0.57% |
| SELL | retest2 | 2025-03-28 09:15:00 | 966.35 | 2025-04-04 15:15:00 | 998.95 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2025-04-01 10:00:00 | 964.70 | 2025-04-04 15:15:00 | 998.95 | STOP_HIT | 1.00 | -3.55% |
| SELL | retest2 | 2025-04-02 10:00:00 | 973.50 | 2025-04-04 15:15:00 | 998.95 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2025-04-02 15:15:00 | 971.95 | 2025-04-04 15:15:00 | 998.95 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2025-06-20 09:15:00 | 1133.60 | 2025-06-20 12:15:00 | 1108.20 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2025-06-23 09:15:00 | 1135.60 | 2025-07-01 14:15:00 | 1249.16 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-23 10:00:00 | 1124.90 | 2025-07-01 14:15:00 | 1237.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-23 14:45:00 | 1125.70 | 2025-07-01 14:15:00 | 1238.27 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-12 09:15:00 | 1277.70 | 2025-08-22 11:15:00 | 1405.47 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-20 10:30:00 | 1282.00 | 2025-10-24 10:15:00 | 1410.20 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-01-06 09:15:00 | 1303.50 | 2026-01-06 11:15:00 | 1322.30 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2026-01-06 10:00:00 | 1301.30 | 2026-01-06 11:15:00 | 1322.30 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2026-01-07 11:45:00 | 1304.10 | 2026-01-07 12:15:00 | 1322.20 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2026-01-12 09:15:00 | 1302.00 | 2026-01-20 12:15:00 | 1236.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 13:45:00 | 1299.60 | 2026-01-20 12:15:00 | 1234.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-12 09:15:00 | 1302.00 | 2026-01-21 09:15:00 | 1344.80 | STOP_HIT | 0.50 | -3.29% |
| SELL | retest2 | 2026-01-19 13:45:00 | 1299.60 | 2026-01-21 09:15:00 | 1344.80 | STOP_HIT | 0.50 | -3.48% |
| SELL | retest2 | 2026-01-28 09:30:00 | 1299.70 | 2026-01-30 09:15:00 | 1340.60 | STOP_HIT | 1.00 | -3.15% |
| SELL | retest2 | 2026-02-01 09:15:00 | 1306.20 | 2026-02-02 12:15:00 | 1240.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-01 09:45:00 | 1304.00 | 2026-02-02 12:15:00 | 1238.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-01 09:15:00 | 1306.20 | 2026-02-03 11:15:00 | 1326.80 | STOP_HIT | 0.50 | -1.58% |
| SELL | retest2 | 2026-02-01 09:45:00 | 1304.00 | 2026-02-03 11:15:00 | 1326.80 | STOP_HIT | 0.50 | -1.75% |
| SELL | retest2 | 2026-02-06 10:15:00 | 1284.90 | 2026-02-26 09:15:00 | 1323.60 | STOP_HIT | 1.00 | -3.01% |
| SELL | retest2 | 2026-02-09 09:45:00 | 1280.90 | 2026-02-26 09:15:00 | 1323.60 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2026-02-09 11:30:00 | 1284.10 | 2026-03-02 09:15:00 | 1228.83 | PARTIAL | 0.50 | 4.30% |
| SELL | retest2 | 2026-02-09 12:30:00 | 1278.50 | 2026-03-02 10:15:00 | 1220.65 | PARTIAL | 0.50 | 4.52% |
| SELL | retest2 | 2026-02-23 14:15:00 | 1290.00 | 2026-03-02 10:15:00 | 1216.86 | PARTIAL | 0.50 | 5.67% |
| SELL | retest2 | 2026-02-23 15:15:00 | 1285.00 | 2026-03-02 10:15:00 | 1219.89 | PARTIAL | 0.50 | 5.07% |
| SELL | retest2 | 2026-02-27 09:30:00 | 1293.50 | 2026-03-02 10:15:00 | 1214.58 | PARTIAL | 0.50 | 6.10% |
| SELL | retest2 | 2026-02-09 11:30:00 | 1284.10 | 2026-03-09 09:15:00 | 1156.41 | TARGET_HIT | 0.50 | 9.94% |
| SELL | retest2 | 2026-02-09 12:30:00 | 1278.50 | 2026-03-09 09:15:00 | 1152.81 | TARGET_HIT | 0.50 | 9.83% |
| SELL | retest2 | 2026-02-23 14:15:00 | 1290.00 | 2026-03-09 09:15:00 | 1155.69 | TARGET_HIT | 0.50 | 10.41% |
| SELL | retest2 | 2026-02-23 15:15:00 | 1285.00 | 2026-03-09 09:15:00 | 1150.65 | TARGET_HIT | 0.50 | 10.46% |
| SELL | retest2 | 2026-02-27 09:30:00 | 1293.50 | 2026-03-09 09:15:00 | 1164.15 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-30 10:30:00 | 1293.70 | 2026-04-30 11:15:00 | 1315.00 | STOP_HIT | 1.00 | -1.65% |
