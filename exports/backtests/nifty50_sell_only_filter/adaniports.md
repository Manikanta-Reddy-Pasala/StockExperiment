# ADANIPORTS (ADANIPORTS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:30:00 (4997 bars)
- **Last close:** 1748.30
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 4 |
| ALERT3 | 7 |
| PENDING | 23 |
| PENDING_CANCEL | 6 |
| ENTRY1 | 0 |
| ENTRY2 | 17 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 15
- **Target hits / Stop hits / Partials:** 0 / 17 / 2
- **Avg / median % per leg:** 0.04% / -1.56%
- **Sum % (uncompounded):** 0.74%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 4 | 22.2% | 0 | 16 | 2 | 0.17% | 3.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 18 | 4 | 22.2% | 0 | 16 | 2 | 0.17% | 3.0% |
| SELL (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.26% | -2.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.26% | -2.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 19 | 4 | 21.1% | 0 | 17 | 2 | 0.04% | 0.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 15:15:00 | 818.70 | 799.57 | 799.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-28 09:15:00 | 823.25 | 802.36 | 801.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-13 09:15:00 | 1262.90 | 1282.85 | 1206.50 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-13 14:15:00 | 1207.35 | 1279.83 | 1206.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 14:15:00 | 1207.35 | 1279.83 | 1206.85 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-03-14 09:15:00 | 1240.75 | 1278.82 | 1207.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-14 10:15:00 | 1251.75 | 1278.55 | 1207.29 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-05-23 14:15:00 | 1439.51 | 1327.12 | 1300.71 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-04 11:15:00 | 1251.75 | 1372.74 | 1332.32 | SL hit qty=0.50 sl=1251.75 alert=retest2 |
| Cross detected — sustain check pending | 2024-06-04 12:15:00 | 1294.95 | 1371.97 | 1332.14 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 13:15:00 | 1319.40 | 1371.45 | 1332.07 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-06-05 09:15:00 | 1188.55 | 1367.53 | 1330.69 | SL hit qty=1.00 sl=1188.55 alert=retest2 |
| Cross detected — sustain check pending | 2024-06-05 10:15:00 | 1311.95 | 1366.98 | 1330.59 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-05 11:15:00 | 1274.65 | 1366.06 | 1330.31 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-06-20 09:15:00 | 1465.85 | 1386.48 | 1351.71 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| CROSSOVER_SKIP | 2024-09-19 15:15:00 | 1410.90 | 1463.18 | 1463.19 | HTF filter: close above htf_sma |
| Stop hit — per-position SL triggered | 2024-11-14 09:15:00 | 1274.65 | 1372.19 | 1400.61 | SL hit qty=0.50 sl=1274.65 alert=retest2 |
| Cross detected — sustain check pending | 2024-12-03 09:15:00 | 1259.95 | 1274.57 | 1333.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 10:15:00 | 1272.00 | 1274.54 | 1333.68 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 11:15:00 | 1301.70 | 1274.81 | 1333.52 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-12-20 13:15:00 | 1188.55 | 1253.94 | 1299.99 | SL hit qty=1.00 sl=1188.55 alert=retest2 |

### Cycle 2 — BUY (started 2025-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 11:15:00 | 1215.20 | 1155.55 | 1155.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 12:15:00 | 1219.80 | 1156.19 | 1155.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 1396.70 | 1397.69 | 1333.28 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 12:15:00 | 1336.70 | 1394.09 | 1338.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 1336.70 | 1394.09 | 1338.72 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-06-20 10:15:00 | 1348.30 | 1391.51 | 1338.79 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-20 11:15:00 | 1344.20 | 1391.04 | 1338.81 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-06-20 14:15:00 | 1349.40 | 1389.62 | 1338.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 15:15:00 | 1349.30 | 1389.22 | 1338.93 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-23 10:15:00 | 1350.40 | 1388.42 | 1339.03 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 11:15:00 | 1358.90 | 1388.13 | 1339.13 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-01 15:15:00 | 1349.80 | 1411.35 | 1390.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 09:15:00 | 1362.80 | 1410.87 | 1390.76 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 3960m) |
| Stop hit — per-position SL triggered | 2025-08-07 10:15:00 | 1332.50 | 1402.28 | 1388.35 | SL hit qty=1.00 sl=1332.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 10:15:00 | 1332.50 | 1402.28 | 1388.35 | SL hit qty=1.00 sl=1332.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 10:15:00 | 1332.50 | 1402.28 | 1388.35 | SL hit qty=1.00 sl=1332.50 alert=retest2 |
| CROSSOVER_SKIP | 2025-08-14 15:15:00 | 1301.40 | 1376.60 | 1376.85 | HTF filter: close above htf_sma |
| Cross detected — sustain check pending | 2025-08-19 10:15:00 | 1350.30 | 1372.20 | 1374.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 11:15:00 | 1357.50 | 1372.05 | 1374.50 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 1356.10 | 1371.90 | 1374.40 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-08-19 14:15:00 | 1369.30 | 1371.70 | 1374.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 15:15:00 | 1369.40 | 1371.67 | 1374.25 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-22 09:15:00 | 1355.10 | 1370.86 | 1373.65 | SL hit qty=1.00 sl=1355.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 1332.50 | 1366.82 | 1371.37 | SL hit qty=1.00 sl=1332.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-09 09:15:00 | 1377.20 | 1350.71 | 1360.65 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 10:15:00 | 1367.00 | 1350.87 | 1360.68 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-17 14:15:00 | 1406.90 | 1368.35 | 1368.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 14:15:00 | 1406.90 | 1368.35 | 1368.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 09:15:00 | 1409.40 | 1369.13 | 1368.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-29 10:15:00 | 1387.90 | 1391.38 | 1381.44 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 11:15:00 | 1380.00 | 1391.27 | 1381.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 1380.00 | 1391.27 | 1381.44 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-09-29 13:15:00 | 1390.20 | 1391.22 | 1381.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-29 14:15:00 | 1382.30 | 1391.13 | 1381.51 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-30 09:15:00 | 1398.80 | 1391.13 | 1381.61 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 10:15:00 | 1398.60 | 1391.20 | 1381.69 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-08 11:15:00 | 1390.80 | 1395.58 | 1385.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 12:15:00 | 1399.50 | 1395.62 | 1385.74 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| CROSSOVER_SKIP | 2026-01-19 15:15:00 | 1403.00 | 1465.15 | 1465.28 | HTF filter: close above htf_sma |
| Stop hit — per-position SL triggered | 2026-01-20 14:15:00 | 1377.60 | 1460.72 | 1463.03 | SL hit qty=1.00 sl=1377.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-20 14:15:00 | 1377.60 | 1460.72 | 1463.03 | SL hit qty=1.00 sl=1377.60 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-22 09:15:00 | 1417.30 | 1453.61 | 1459.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 10:15:00 | 1412.00 | 1453.19 | 1459.06 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-23 12:15:00 | 1377.60 | 1448.47 | 1456.39 | SL hit qty=1.00 sl=1377.60 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-29 12:15:00 | 1390.50 | 1432.23 | 1446.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 13:15:00 | 1399.90 | 1431.91 | 1446.75 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 1417.10 | 1431.76 | 1446.61 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-01-30 09:15:00 | 1422.40 | 1431.53 | 1446.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:15:00 | 1423.80 | 1431.45 | 1446.23 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-30 13:15:00 | 1422.30 | 1431.11 | 1445.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-30 14:15:00 | 1418.70 | 1430.99 | 1445.70 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-30 15:15:00 | 1419.80 | 1430.87 | 1445.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-02 09:15:00 | 1381.60 | 1430.38 | 1445.25 | ENTRY2 sustain failed after 3960m |
| Stop hit — per-position SL triggered | 2026-02-02 09:15:00 | 1377.60 | 1430.38 | 1445.25 | SL hit qty=1.00 sl=1377.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-02 09:15:00 | 1397.70 | 1430.38 | 1445.25 | SL hit qty=1.00 sl=1397.70 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-03 09:15:00 | 1502.30 | 1428.77 | 1443.91 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 10:15:00 | 1505.80 | 1429.54 | 1444.22 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-09 09:15:00 | 1562.70 | 1457.76 | 1457.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 1562.70 | 1457.76 | 1457.39 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 1362.40 | 1476.40 | 1476.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 1352.40 | 1448.69 | 1461.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1457.10 | 1403.09 | 1430.63 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 1457.10 | 1403.09 | 1430.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1457.10 | 1403.09 | 1430.63 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-08 11:15:00 | 1452.50 | 1404.14 | 1430.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-08 12:15:00 | 1456.20 | 1404.65 | 1431.00 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-08 13:15:00 | 1453.50 | 1405.14 | 1431.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 14:15:00 | 1452.70 | 1405.61 | 1431.22 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 1454.40 | 1413.21 | 1433.19 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-13 10:15:00 | 1461.70 | 1413.69 | 1433.33 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 1485.60 | 1417.30 | 1434.58 | SL hit qty=1.00 sl=1485.60 alert=retest2 |

### Cycle 6 — BUY (started 2026-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 12:15:00 | 1600.80 | 1450.09 | 1449.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 09:15:00 | 1629.40 | 1481.25 | 1466.34 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-03-14 10:15:00 | 1251.75 | 2024-05-23 14:15:00 | 1439.51 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2024-03-14 10:15:00 | 1251.75 | 2024-06-04 11:15:00 | 1251.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest2 | 2024-06-04 13:15:00 | 1319.40 | 2024-06-05 09:15:00 | 1188.55 | STOP_HIT | 1.00 | -9.92% |
| BUY | retest2 | 2024-06-05 11:15:00 | 1274.65 | 2024-06-20 09:15:00 | 1465.85 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2024-06-05 11:15:00 | 1274.65 | 2024-11-14 09:15:00 | 1274.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest2 | 2024-12-03 10:15:00 | 1272.00 | 2024-12-20 13:15:00 | 1188.55 | STOP_HIT | 1.00 | -6.56% |
| BUY | retest2 | 2025-06-20 15:15:00 | 1349.30 | 2025-08-07 10:15:00 | 1332.50 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-06-23 11:15:00 | 1358.90 | 2025-08-07 10:15:00 | 1332.50 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-08-04 09:15:00 | 1362.80 | 2025-08-07 10:15:00 | 1332.50 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-08-19 11:15:00 | 1357.50 | 2025-08-22 09:15:00 | 1355.10 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2025-08-19 15:15:00 | 1369.40 | 2025-08-26 09:15:00 | 1332.50 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2025-09-09 10:15:00 | 1367.00 | 2025-09-17 14:15:00 | 1406.90 | STOP_HIT | 1.00 | 2.92% |
| BUY | retest2 | 2025-09-30 10:15:00 | 1398.60 | 2026-01-20 14:15:00 | 1377.60 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-10-08 12:15:00 | 1399.50 | 2026-01-20 14:15:00 | 1377.60 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2026-01-22 10:15:00 | 1412.00 | 2026-01-23 12:15:00 | 1377.60 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2026-01-29 13:15:00 | 1399.90 | 2026-02-02 09:15:00 | 1377.60 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2026-01-30 10:15:00 | 1423.80 | 2026-02-02 09:15:00 | 1397.70 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2026-02-03 10:15:00 | 1505.80 | 2026-02-09 09:15:00 | 1562.70 | STOP_HIT | 1.00 | 3.78% |
| SELL | retest2 | 2026-04-08 14:15:00 | 1452.70 | 2026-04-15 09:15:00 | 1485.60 | STOP_HIT | 1.00 | -2.26% |
