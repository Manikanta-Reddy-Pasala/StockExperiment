# Mahanagar Gas Ltd. (MGL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1173.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 11 |
| ALERT2 | 10 |
| ALERT2_SKIP | 6 |
| ALERT3 | 39 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 36 |
| PARTIAL | 5 |
| TARGET_HIT | 4 |
| STOP_HIT | 32 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 41 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 29
- **Target hits / Stop hits / Partials:** 4 / 32 / 5
- **Avg / median % per leg:** -0.59% / -2.21%
- **Sum % (uncompounded):** -24.09%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 2 | 14.3% | 2 | 12 | 0 | -0.98% | -13.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 14 | 2 | 14.3% | 2 | 12 | 0 | -0.98% | -13.7% |
| SELL (all) | 27 | 10 | 37.0% | 2 | 20 | 5 | -0.38% | -10.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 27 | 10 | 37.0% | 2 | 20 | 5 | -0.38% | -10.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 41 | 12 | 29.3% | 4 | 32 | 5 | -0.59% | -24.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 09:15:00 | 980.70 | 1054.00 | 1054.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-18 12:15:00 | 975.60 | 1051.79 | 1053.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-04 12:15:00 | 1035.00 | 1031.05 | 1039.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-04 13:00:00 | 1035.00 | 1031.05 | 1039.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 09:15:00 | 1041.25 | 1031.19 | 1039.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-05 10:00:00 | 1041.25 | 1031.19 | 1039.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 10:15:00 | 1039.80 | 1031.28 | 1039.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-05 12:30:00 | 1030.50 | 1031.34 | 1039.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-05 15:00:00 | 1032.55 | 1031.39 | 1039.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-06 09:15:00 | 1061.85 | 1031.68 | 1039.66 | SL hit (close>static) qty=1.00 sl=1041.30 alert=retest2 |

### Cycle 2 — BUY (started 2023-10-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-04 14:15:00 | 1123.10 | 1042.75 | 1042.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-13 09:15:00 | 1140.80 | 1067.43 | 1056.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-20 09:15:00 | 1044.45 | 1085.08 | 1067.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-20 09:15:00 | 1044.45 | 1085.08 | 1067.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 09:15:00 | 1044.45 | 1085.08 | 1067.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-20 10:00:00 | 1044.45 | 1085.08 | 1067.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 10:15:00 | 1040.40 | 1084.64 | 1067.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-20 12:15:00 | 1050.80 | 1084.13 | 1067.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-20 12:15:00 | 1030.90 | 1083.60 | 1067.17 | SL hit (close<static) qty=1.00 sl=1032.25 alert=retest2 |

### Cycle 3 — SELL (started 2023-11-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-02 09:15:00 | 1026.70 | 1054.56 | 1054.57 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-06 09:15:00 | 1116.85 | 1051.06 | 1050.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-07 09:15:00 | 1124.75 | 1055.51 | 1053.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-06 09:15:00 | 1342.00 | 1464.67 | 1379.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 09:15:00 | 1342.00 | 1464.67 | 1379.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 1342.00 | 1464.67 | 1379.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 10:00:00 | 1342.00 | 1464.67 | 1379.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 10:15:00 | 1326.95 | 1463.30 | 1379.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 11:00:00 | 1326.95 | 1463.30 | 1379.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 11:15:00 | 1351.30 | 1365.22 | 1348.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-26 13:15:00 | 1356.15 | 1365.09 | 1348.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 11:15:00 | 1354.60 | 1365.12 | 1349.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-26 11:15:00 | 1491.77 | 1409.54 | 1385.60 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2024-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 12:15:00 | 1301.00 | 1378.02 | 1378.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-17 11:15:00 | 1298.00 | 1373.76 | 1375.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 09:15:00 | 1374.60 | 1335.82 | 1353.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 09:15:00 | 1374.60 | 1335.82 | 1353.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 1374.60 | 1335.82 | 1353.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:45:00 | 1379.90 | 1335.82 | 1353.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 10:15:00 | 1381.75 | 1336.28 | 1353.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 11:30:00 | 1366.30 | 1336.56 | 1353.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 09:15:00 | 1319.45 | 1338.01 | 1353.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 10:15:00 | 1297.98 | 1337.22 | 1353.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 11:15:00 | 1253.48 | 1336.26 | 1352.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-06-04 12:15:00 | 1229.67 | 1335.42 | 1351.99 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 6 — BUY (started 2024-06-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-14 10:15:00 | 1467.00 | 1362.93 | 1362.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 10:15:00 | 1480.00 | 1369.83 | 1366.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-14 13:15:00 | 1745.30 | 1748.64 | 1652.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-14 13:45:00 | 1744.85 | 1748.64 | 1652.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 1844.70 | 1877.03 | 1812.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 09:30:00 | 1860.95 | 1871.83 | 1813.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 14:15:00 | 1854.00 | 1871.55 | 1814.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 09:45:00 | 1861.50 | 1871.26 | 1815.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 11:00:00 | 1852.30 | 1871.08 | 1815.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 1815.90 | 1868.72 | 1816.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 09:45:00 | 1812.90 | 1868.72 | 1816.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 1807.75 | 1868.11 | 1816.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 11:00:00 | 1807.75 | 1868.11 | 1816.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 11:15:00 | 1799.55 | 1867.43 | 1816.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 11:30:00 | 1797.20 | 1867.43 | 1816.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-14 12:15:00 | 1794.00 | 1866.70 | 1815.97 | SL hit (close<static) qty=1.00 sl=1797.30 alert=retest2 |

### Cycle 7 — SELL (started 2024-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 15:15:00 | 1577.70 | 1779.89 | 1780.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 10:15:00 | 1563.95 | 1775.80 | 1778.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 10:15:00 | 1285.45 | 1282.78 | 1378.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-31 10:45:00 | 1284.15 | 1282.78 | 1378.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 1301.05 | 1282.64 | 1329.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:45:00 | 1328.75 | 1282.64 | 1329.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 1317.85 | 1282.49 | 1327.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 09:45:00 | 1258.80 | 1307.85 | 1329.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:30:00 | 1261.10 | 1306.33 | 1326.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 13:45:00 | 1260.30 | 1300.25 | 1322.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-20 13:15:00 | 1345.20 | 1300.71 | 1320.79 | SL hit (close>static) qty=1.00 sl=1336.95 alert=retest2 |

### Cycle 8 — BUY (started 2025-03-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-26 12:15:00 | 1366.20 | 1318.69 | 1318.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 09:15:00 | 1387.90 | 1323.07 | 1320.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 13:15:00 | 1329.85 | 1338.10 | 1329.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 13:15:00 | 1329.85 | 1338.10 | 1329.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 13:15:00 | 1329.85 | 1338.10 | 1329.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 13:45:00 | 1330.95 | 1338.10 | 1329.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 14:15:00 | 1331.65 | 1338.04 | 1329.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 15:15:00 | 1335.05 | 1338.04 | 1329.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 1281.25 | 1337.45 | 1329.15 | SL hit (close<static) qty=1.00 sl=1321.05 alert=retest2 |

### Cycle 9 — SELL (started 2025-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-16 11:15:00 | 1250.10 | 1321.89 | 1322.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-17 09:15:00 | 1245.20 | 1318.48 | 1320.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-22 09:15:00 | 1332.50 | 1312.65 | 1317.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-22 09:15:00 | 1332.50 | 1312.65 | 1317.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 09:15:00 | 1332.50 | 1312.65 | 1317.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-22 09:45:00 | 1330.20 | 1312.65 | 1317.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 10:15:00 | 1326.00 | 1312.78 | 1317.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-23 10:30:00 | 1309.00 | 1314.27 | 1317.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-23 11:30:00 | 1315.10 | 1314.24 | 1317.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-24 10:15:00 | 1337.50 | 1314.88 | 1318.02 | SL hit (close>static) qty=1.00 sl=1333.30 alert=retest2 |

### Cycle 10 — BUY (started 2025-05-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 14:15:00 | 1360.00 | 1320.22 | 1320.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-02 15:15:00 | 1360.70 | 1320.62 | 1320.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 12:15:00 | 1330.60 | 1339.45 | 1330.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 12:15:00 | 1330.60 | 1339.45 | 1330.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 12:15:00 | 1330.60 | 1339.45 | 1330.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 13:00:00 | 1330.60 | 1339.45 | 1330.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 13:15:00 | 1362.40 | 1339.68 | 1330.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 15:15:00 | 1364.00 | 1339.91 | 1331.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 09:15:00 | 1321.20 | 1358.90 | 1345.91 | SL hit (close<static) qty=1.00 sl=1328.30 alert=retest2 |

### Cycle 11 — SELL (started 2025-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 11:15:00 | 1285.60 | 1407.91 | 1408.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 13:15:00 | 1274.70 | 1357.25 | 1377.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 1331.60 | 1316.51 | 1345.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-16 10:00:00 | 1331.60 | 1316.51 | 1345.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 1350.00 | 1316.84 | 1345.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:00:00 | 1350.00 | 1316.84 | 1345.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 1354.70 | 1317.22 | 1345.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 11:30:00 | 1340.60 | 1322.36 | 1345.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 09:30:00 | 1346.70 | 1323.35 | 1345.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 09:30:00 | 1344.00 | 1324.58 | 1345.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 13:15:00 | 1279.37 | 1321.94 | 1342.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 15:15:00 | 1276.80 | 1321.07 | 1341.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 09:15:00 | 1273.57 | 1320.58 | 1341.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-07 10:15:00 | 1304.40 | 1303.05 | 1326.88 | SL hit (close>ema200) qty=0.50 sl=1303.05 alert=retest2 |

### Cycle 12 — BUY (started 2026-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 14:15:00 | 1150.90 | 1081.30 | 1080.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 1154.50 | 1092.50 | 1086.92 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-09-05 12:30:00 | 1030.50 | 2023-09-06 09:15:00 | 1061.85 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2023-09-05 15:00:00 | 1032.55 | 2023-09-06 09:15:00 | 1061.85 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2023-09-12 09:30:00 | 1030.40 | 2023-09-12 10:15:00 | 1044.20 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2023-09-12 11:45:00 | 1033.90 | 2023-09-14 11:15:00 | 1043.40 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2023-09-21 12:00:00 | 1031.95 | 2023-10-03 09:15:00 | 1086.00 | STOP_HIT | 1.00 | -5.24% |
| SELL | retest2 | 2023-09-26 10:15:00 | 1031.70 | 2023-10-03 09:15:00 | 1086.00 | STOP_HIT | 1.00 | -5.26% |
| SELL | retest2 | 2023-09-26 13:30:00 | 1031.20 | 2023-10-03 09:15:00 | 1086.00 | STOP_HIT | 1.00 | -5.31% |
| BUY | retest2 | 2023-10-20 12:15:00 | 1050.80 | 2023-10-20 12:15:00 | 1030.90 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2024-03-26 13:15:00 | 1356.15 | 2024-04-26 11:15:00 | 1491.77 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-28 11:15:00 | 1354.60 | 2024-04-26 11:15:00 | 1490.06 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-08 09:15:00 | 1355.55 | 2024-05-09 10:15:00 | 1345.45 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-05-09 11:15:00 | 1357.25 | 2024-05-09 12:15:00 | 1343.50 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2024-06-03 11:30:00 | 1366.30 | 2024-06-04 10:15:00 | 1297.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 09:15:00 | 1319.45 | 2024-06-04 11:15:00 | 1253.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 11:30:00 | 1366.30 | 2024-06-04 12:15:00 | 1229.67 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-04 09:15:00 | 1319.45 | 2024-06-04 12:15:00 | 1187.51 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-06 10:15:00 | 1367.50 | 2024-06-07 11:15:00 | 1397.70 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2024-06-06 11:30:00 | 1360.80 | 2024-06-07 11:15:00 | 1397.70 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2024-10-10 09:30:00 | 1860.95 | 2024-10-14 12:15:00 | 1794.00 | STOP_HIT | 1.00 | -3.60% |
| BUY | retest2 | 2024-10-10 14:15:00 | 1854.00 | 2024-10-14 12:15:00 | 1794.00 | STOP_HIT | 1.00 | -3.24% |
| BUY | retest2 | 2024-10-11 09:45:00 | 1861.50 | 2024-10-14 12:15:00 | 1794.00 | STOP_HIT | 1.00 | -3.63% |
| BUY | retest2 | 2024-10-11 11:00:00 | 1852.30 | 2024-10-14 12:15:00 | 1794.00 | STOP_HIT | 1.00 | -3.15% |
| SELL | retest2 | 2025-02-12 09:45:00 | 1258.80 | 2025-02-20 13:15:00 | 1345.20 | STOP_HIT | 1.00 | -6.86% |
| SELL | retest2 | 2025-02-14 10:30:00 | 1261.10 | 2025-02-20 13:15:00 | 1345.20 | STOP_HIT | 1.00 | -6.67% |
| SELL | retest2 | 2025-02-17 13:45:00 | 1260.30 | 2025-02-20 13:15:00 | 1345.20 | STOP_HIT | 1.00 | -6.74% |
| SELL | retest2 | 2025-02-28 11:30:00 | 1253.40 | 2025-03-06 14:15:00 | 1338.45 | STOP_HIT | 1.00 | -6.79% |
| BUY | retest2 | 2025-04-04 15:15:00 | 1335.05 | 2025-04-07 09:15:00 | 1281.25 | STOP_HIT | 1.00 | -4.03% |
| SELL | retest2 | 2025-04-23 10:30:00 | 1309.00 | 2025-04-24 10:15:00 | 1337.50 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-04-23 11:30:00 | 1315.10 | 2025-04-24 10:15:00 | 1337.50 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-04-25 09:30:00 | 1300.80 | 2025-04-28 14:15:00 | 1335.70 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2025-04-28 10:15:00 | 1313.90 | 2025-04-28 14:15:00 | 1335.70 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-05-09 15:15:00 | 1364.00 | 2025-05-27 09:15:00 | 1321.20 | STOP_HIT | 1.00 | -3.14% |
| BUY | retest2 | 2025-05-28 11:00:00 | 1363.40 | 2025-05-30 14:15:00 | 1323.80 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2025-05-29 09:30:00 | 1371.60 | 2025-05-30 14:15:00 | 1323.80 | STOP_HIT | 1.00 | -3.48% |
| BUY | retest2 | 2025-06-02 09:30:00 | 1366.40 | 2025-06-02 12:15:00 | 1326.60 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2025-09-18 11:30:00 | 1340.60 | 2025-09-24 13:15:00 | 1279.37 | PARTIAL | 0.50 | 4.57% |
| SELL | retest2 | 2025-09-19 09:30:00 | 1346.70 | 2025-09-24 15:15:00 | 1276.80 | PARTIAL | 0.50 | 5.19% |
| SELL | retest2 | 2025-09-22 09:30:00 | 1344.00 | 2025-09-25 09:15:00 | 1273.57 | PARTIAL | 0.50 | 5.24% |
| SELL | retest2 | 2025-09-18 11:30:00 | 1340.60 | 2025-10-07 10:15:00 | 1304.40 | STOP_HIT | 0.50 | 2.70% |
| SELL | retest2 | 2025-09-19 09:30:00 | 1346.70 | 2025-10-07 10:15:00 | 1304.40 | STOP_HIT | 0.50 | 3.14% |
| SELL | retest2 | 2025-09-22 09:30:00 | 1344.00 | 2025-10-07 10:15:00 | 1304.40 | STOP_HIT | 0.50 | 2.95% |
