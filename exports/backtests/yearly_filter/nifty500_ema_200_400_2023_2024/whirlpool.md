# Whirlpool of India Ltd. (WHIRLPOOL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 954.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 1 |
| ALERT3 | 42 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 25 |
| PARTIAL | 4 |
| TARGET_HIT | 4 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 29 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 19
- **Target hits / Stop hits / Partials:** 4 / 21 / 4
- **Avg / median % per leg:** 0.50% / -1.66%
- **Sum % (uncompounded):** 14.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 2 | 18.2% | 2 | 9 | 0 | 0.25% | 2.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 11 | 2 | 18.2% | 2 | 9 | 0 | 0.25% | 2.7% |
| SELL (all) | 18 | 8 | 44.4% | 2 | 12 | 4 | 0.66% | 11.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 18 | 8 | 44.4% | 2 | 12 | 4 | 0.66% | 11.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 29 | 10 | 34.5% | 4 | 21 | 4 | 0.50% | 14.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 15:15:00 | 1565.95 | 1602.03 | 1602.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-28 09:15:00 | 1548.55 | 1601.50 | 1601.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 12:15:00 | 1392.00 | 1382.16 | 1431.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-18 13:00:00 | 1392.00 | 1382.16 | 1431.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 11:15:00 | 1321.10 | 1264.50 | 1301.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-03 12:00:00 | 1321.10 | 1264.50 | 1301.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 12:15:00 | 1333.90 | 1265.20 | 1301.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-03 13:00:00 | 1333.90 | 1265.20 | 1301.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2024-04-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 09:15:00 | 1431.05 | 1326.93 | 1326.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-19 14:15:00 | 1440.00 | 1338.02 | 1332.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-07 11:15:00 | 1414.45 | 1417.09 | 1382.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-07 11:30:00 | 1412.75 | 1417.09 | 1382.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 1425.70 | 1479.36 | 1439.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:00:00 | 1425.70 | 1479.36 | 1439.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 1452.00 | 1479.09 | 1439.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:30:00 | 1410.00 | 1479.09 | 1439.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 2075.70 | 2110.37 | 2023.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 09:45:00 | 2060.75 | 2110.37 | 2023.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 1983.15 | 2106.64 | 2024.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:00:00 | 1983.15 | 2106.64 | 2024.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 2003.10 | 2105.60 | 2024.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 11:30:00 | 2019.25 | 2104.88 | 2024.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 11:30:00 | 2010.00 | 2092.72 | 2025.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-27 13:15:00 | 2211.00 | 2094.59 | 2037.82 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 15:15:00 | 1788.65 | 2129.01 | 2129.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 1749.75 | 2125.24 | 2127.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 14:15:00 | 1929.20 | 1928.69 | 1995.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-10 15:00:00 | 1929.20 | 1928.69 | 1995.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 1976.95 | 1928.87 | 1979.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 10:45:00 | 1978.45 | 1928.87 | 1979.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 10:15:00 | 1138.90 | 1061.91 | 1145.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-24 10:45:00 | 1148.55 | 1061.91 | 1145.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 13:15:00 | 1161.50 | 1064.31 | 1145.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-24 13:30:00 | 1189.80 | 1064.31 | 1145.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 14:15:00 | 1188.55 | 1065.54 | 1145.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-24 14:30:00 | 1207.00 | 1065.54 | 1145.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 1140.40 | 1068.22 | 1145.59 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2025-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 10:15:00 | 1304.20 | 1188.53 | 1188.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 1357.00 | 1232.76 | 1216.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 12:15:00 | 1339.40 | 1341.08 | 1296.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-08 12:45:00 | 1340.90 | 1341.08 | 1296.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 1343.10 | 1379.09 | 1340.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:45:00 | 1341.20 | 1379.09 | 1340.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 11:15:00 | 1345.80 | 1378.76 | 1340.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 11:45:00 | 1343.40 | 1378.76 | 1340.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 13:15:00 | 1340.00 | 1378.05 | 1340.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 14:00:00 | 1340.00 | 1378.05 | 1340.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 1334.50 | 1377.61 | 1340.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 14:30:00 | 1323.90 | 1377.61 | 1340.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 1336.00 | 1377.20 | 1340.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:15:00 | 1339.90 | 1377.20 | 1340.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 1323.90 | 1376.17 | 1339.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:45:00 | 1323.00 | 1376.17 | 1339.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 12:15:00 | 1340.00 | 1375.43 | 1339.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 13:00:00 | 1340.00 | 1375.43 | 1339.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 1332.60 | 1375.00 | 1339.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 14:00:00 | 1332.60 | 1375.00 | 1339.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 1313.90 | 1374.39 | 1339.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:00:00 | 1313.90 | 1374.39 | 1339.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 1339.60 | 1372.66 | 1339.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:45:00 | 1336.50 | 1372.66 | 1339.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 1340.00 | 1372.34 | 1339.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 1340.00 | 1372.34 | 1339.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 1338.40 | 1372.00 | 1339.51 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 1287.00 | 1319.14 | 1319.22 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 10:15:00 | 1369.60 | 1318.68 | 1318.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 09:15:00 | 1378.00 | 1323.82 | 1321.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 14:15:00 | 1328.50 | 1332.85 | 1326.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 14:15:00 | 1328.50 | 1332.85 | 1326.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 1328.50 | 1332.85 | 1326.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:45:00 | 1330.00 | 1332.85 | 1326.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 1338.50 | 1332.91 | 1326.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 10:15:00 | 1347.10 | 1332.91 | 1326.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 12:45:00 | 1345.70 | 1334.10 | 1327.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 15:15:00 | 1355.00 | 1334.18 | 1327.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 12:15:00 | 1344.40 | 1337.14 | 1329.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 1331.30 | 1337.33 | 1329.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:30:00 | 1328.40 | 1337.33 | 1329.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 1326.70 | 1337.22 | 1329.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:30:00 | 1329.60 | 1337.22 | 1329.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 1336.00 | 1337.21 | 1329.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:30:00 | 1329.10 | 1337.21 | 1329.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 1333.50 | 1337.14 | 1329.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 14:45:00 | 1345.40 | 1336.99 | 1330.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 09:30:00 | 1345.80 | 1337.10 | 1330.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 14:15:00 | 1325.90 | 1336.84 | 1330.18 | SL hit (close<static) qty=1.00 sl=1326.10 alert=retest2 |

### Cycle 7 — SELL (started 2025-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 14:15:00 | 1233.50 | 1324.06 | 1324.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 10:15:00 | 1222.10 | 1321.18 | 1322.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 14:15:00 | 1239.70 | 1238.29 | 1271.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-16 15:00:00 | 1239.70 | 1238.29 | 1271.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 1321.30 | 1239.15 | 1272.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:45:00 | 1314.90 | 1239.15 | 1272.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 1331.50 | 1240.07 | 1272.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:30:00 | 1334.60 | 1240.07 | 1272.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2025-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 10:15:00 | 1407.30 | 1297.64 | 1297.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 14:15:00 | 1423.00 | 1302.06 | 1299.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 09:15:00 | 1314.00 | 1320.43 | 1309.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-07 10:00:00 | 1314.00 | 1320.43 | 1309.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 1306.40 | 1320.29 | 1309.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:00:00 | 1306.40 | 1320.29 | 1309.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 1315.30 | 1320.24 | 1309.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 13:15:00 | 1338.90 | 1320.19 | 1309.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 09:15:00 | 1304.00 | 1320.61 | 1310.32 | SL hit (close<static) qty=1.00 sl=1306.30 alert=retest2 |

### Cycle 9 — SELL (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 10:15:00 | 1252.20 | 1301.47 | 1301.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 15:15:00 | 1241.30 | 1298.69 | 1300.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 10:15:00 | 839.85 | 838.07 | 924.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 10:30:00 | 840.10 | 838.07 | 924.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 898.25 | 843.11 | 922.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 14:15:00 | 890.15 | 845.30 | 921.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 12:00:00 | 891.80 | 851.91 | 920.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 14:45:00 | 889.35 | 853.10 | 920.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 09:15:00 | 888.75 | 856.52 | 919.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 915.85 | 860.00 | 918.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 916.55 | 860.00 | 918.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 926.00 | 861.21 | 918.25 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-17 12:15:00 | 926.00 | 861.21 | 918.25 | SL hit (close>static) qty=1.00 sl=922.40 alert=retest2 |

### Cycle 10 — BUY (started 2026-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 14:15:00 | 996.50 | 884.31 | 883.77 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-09-17 11:30:00 | 2019.25 | 2024-09-27 13:15:00 | 2211.00 | TARGET_HIT | 1.00 | 9.50% |
| BUY | retest2 | 2024-09-20 11:30:00 | 2010.00 | 2024-09-30 09:15:00 | 2221.18 | TARGET_HIT | 1.00 | 10.51% |
| BUY | retest2 | 2024-10-31 13:00:00 | 2008.35 | 2024-11-11 09:15:00 | 1973.25 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2024-11-01 18:00:00 | 2006.50 | 2024-11-11 09:15:00 | 1973.25 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-09-12 10:15:00 | 1347.10 | 2025-09-22 14:15:00 | 1325.90 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-09-15 12:45:00 | 1345.70 | 2025-09-22 14:15:00 | 1325.90 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-09-15 15:15:00 | 1355.00 | 2025-09-23 15:15:00 | 1320.00 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2025-09-17 12:15:00 | 1344.40 | 2025-09-23 15:15:00 | 1320.00 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-09-19 14:45:00 | 1345.40 | 2025-09-23 15:15:00 | 1320.00 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-09-22 09:30:00 | 1345.80 | 2025-09-23 15:15:00 | 1320.00 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-11-07 13:15:00 | 1338.90 | 2025-11-10 09:15:00 | 1304.00 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2026-02-10 14:15:00 | 890.15 | 2026-02-17 12:15:00 | 926.00 | STOP_HIT | 1.00 | -4.03% |
| SELL | retest2 | 2026-02-12 12:00:00 | 891.80 | 2026-02-17 12:15:00 | 926.00 | STOP_HIT | 1.00 | -3.83% |
| SELL | retest2 | 2026-02-12 14:45:00 | 889.35 | 2026-02-17 12:15:00 | 926.00 | STOP_HIT | 1.00 | -4.12% |
| SELL | retest2 | 2026-02-16 09:15:00 | 888.75 | 2026-02-17 12:15:00 | 926.00 | STOP_HIT | 1.00 | -4.19% |
| SELL | retest2 | 2026-02-19 12:00:00 | 912.10 | 2026-02-20 10:15:00 | 930.50 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2026-02-20 10:00:00 | 915.15 | 2026-02-20 10:15:00 | 930.50 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2026-02-24 12:15:00 | 916.55 | 2026-02-27 13:15:00 | 921.15 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2026-02-24 13:15:00 | 908.70 | 2026-02-27 13:15:00 | 921.15 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2026-02-26 12:30:00 | 897.50 | 2026-02-27 13:15:00 | 921.15 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2026-02-26 14:15:00 | 900.80 | 2026-03-09 09:15:00 | 870.72 | PARTIAL | 0.50 | 3.34% |
| SELL | retest2 | 2026-02-26 14:45:00 | 900.50 | 2026-03-09 09:15:00 | 863.26 | PARTIAL | 0.50 | 4.13% |
| SELL | retest2 | 2026-02-26 14:15:00 | 900.80 | 2026-03-11 09:15:00 | 895.80 | STOP_HIT | 0.50 | 0.56% |
| SELL | retest2 | 2026-02-26 14:45:00 | 900.50 | 2026-03-11 09:15:00 | 895.80 | STOP_HIT | 0.50 | 0.52% |
| SELL | retest2 | 2026-03-02 10:00:00 | 900.45 | 2026-03-12 15:15:00 | 855.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 12:15:00 | 898.00 | 2026-03-13 09:15:00 | 853.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 10:00:00 | 900.45 | 2026-03-16 10:15:00 | 810.41 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-06 12:15:00 | 898.00 | 2026-03-19 14:15:00 | 808.20 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-22 09:15:00 | 912.45 | 2026-04-22 09:15:00 | 933.25 | STOP_HIT | 1.00 | -2.28% |
