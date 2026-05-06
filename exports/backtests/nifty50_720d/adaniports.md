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
| CROSSOVER | 10 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT2_SKIP | 6 |
| ALERT3 | 7 |
| PENDING | 19 |
| PENDING_CANCEL | 5 |
| ENTRY1 | 1 |
| ENTRY2 | 13 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 10
- **Target hits / Stop hits / Partials:** 0 / 14 / 3
- **Avg / median % per leg:** 2.36% / -0.97%
- **Sum % (uncompounded):** 40.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 5 | 50.0% | 0 | 8 | 2 | 2.59% | 25.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 10 | 5 | 50.0% | 0 | 8 | 2 | 2.59% | 25.9% |
| SELL (all) | 7 | 2 | 28.6% | 0 | 6 | 1 | 2.02% | 14.1% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 16.10% | 32.2% |
| SELL @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -3.61% | -18.1% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 1 | 1 | 16.10% | 32.2% |
| retest2 (combined) | 15 | 5 | 33.3% | 0 | 13 | 2 | 0.52% | 7.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-02 09:15:00 | 774.15 | 799.48 | 799.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-02 14:15:00 | 772.40 | 798.39 | 798.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-06 09:15:00 | 803.45 | 797.89 | 798.69 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-06 09:15:00 | 803.45 | 797.89 | 798.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 09:15:00 | 803.45 | 797.89 | 798.69 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2023-11-07 13:15:00 | 796.95 | 798.31 | 798.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-07 14:15:00 | 796.95 | 798.30 | 798.85 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2023-11-08 09:15:00 | 804.70 | 798.40 | 798.90 | SL hit qty=1.00 sl=804.70 alert=retest2 |

### Cycle 2 — BUY (started 2023-11-08 15:15:00)

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
| Stop hit — per-position SL triggered | 2024-09-19 15:15:00 | 1410.90 | 1463.18 | 1463.19 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |

### Cycle 3 — SELL (started 2024-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 15:15:00 | 1410.90 | 1463.18 | 1463.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 09:15:00 | 1385.90 | 1454.86 | 1458.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 10:15:00 | 1411.00 | 1399.66 | 1422.94 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-10-31 10:15:00 | 1383.70 | 1399.55 | 1422.08 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-31 11:15:00 | 1384.50 | 1399.40 | 1421.90 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-11-21 09:15:00 | 1176.83 | 1353.97 | 1388.20 | Partial book 0.50 @ 15%; trail SL->entry alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 12:15:00 | 1142.10 | 1102.39 | 1141.14 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-03-12 11:15:00 | 1109.40 | 1111.87 | 1141.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-03-12 12:15:00 | 1122.25 | 1111.98 | 1141.24 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-03-13 14:15:00 | 1119.55 | 1113.20 | 1140.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 15:15:00 | 1119.20 | 1113.26 | 1140.47 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-03-17 09:15:00 | 1146.40 | 1113.54 | 1140.47 | SL hit qty=1.00 sl=1146.40 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-07 09:15:00 | 1085.85 | 1154.72 | 1155.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 10:15:00 | 1105.15 | 1154.23 | 1155.18 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-04-08 09:15:00 | 1146.40 | 1151.45 | 1153.74 | SL hit qty=1.00 sl=1146.40 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-09 09:15:00 | 1114.20 | 1149.82 | 1152.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 10:15:00 | 1118.35 | 1149.50 | 1152.66 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 1146.40 | 1148.72 | 1152.16 | SL hit qty=1.00 sl=1146.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-16 11:15:00 | 1215.20 | 1155.55 | 1155.41 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest1 |

### Cycle 4 — BUY (started 2025-04-16 11:15:00)

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

### Cycle 5 — SELL (started 2025-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 15:15:00 | 1301.40 | 1376.60 | 1376.85 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-09-17 14:15:00)

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
| Stop hit — per-position SL triggered | 2026-01-19 15:15:00 | 1403.00 | 1465.15 | 1465.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 15:15:00 | 1403.00 | 1465.15 | 1465.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2026-01-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 15:15:00 | 1403.00 | 1465.15 | 1465.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 10:15:00 | 1395.90 | 1463.86 | 1464.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 1502.30 | 1428.77 | 1443.91 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 1502.30 | 1428.77 | 1443.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1502.30 | 1428.77 | 1443.91 | EMA400 retest candle locked |

### Cycle 8 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 1562.70 | 1457.76 | 1457.39 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2026-03-13 11:15:00)

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

### Cycle 10 — BUY (started 2026-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 12:15:00 | 1600.80 | 1450.09 | 1449.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 09:15:00 | 1629.40 | 1481.25 | 1466.34 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-11-07 14:15:00 | 796.95 | 2023-11-08 09:15:00 | 804.70 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-03-14 10:15:00 | 1251.75 | 2024-05-23 14:15:00 | 1439.51 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2024-03-14 10:15:00 | 1251.75 | 2024-06-04 11:15:00 | 1251.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest2 | 2024-06-04 13:15:00 | 1319.40 | 2024-06-05 09:15:00 | 1188.55 | STOP_HIT | 1.00 | -9.92% |
| BUY | retest2 | 2024-06-05 11:15:00 | 1274.65 | 2024-06-20 09:15:00 | 1465.85 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2024-06-05 11:15:00 | 1274.65 | 2024-09-19 15:15:00 | 1410.90 | STOP_HIT | 0.50 | 10.69% |
| SELL | retest1 | 2024-10-31 11:15:00 | 1384.50 | 2024-11-21 09:15:00 | 1176.83 | PARTIAL | 0.50 | 15.00% |
| SELL | retest1 | 2024-10-31 11:15:00 | 1384.50 | 2025-03-17 09:15:00 | 1146.40 | STOP_HIT | 0.50 | 17.20% |
| SELL | retest2 | 2025-03-13 15:15:00 | 1119.20 | 2025-04-08 09:15:00 | 1146.40 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2025-04-07 10:15:00 | 1105.15 | 2025-04-11 09:15:00 | 1146.40 | STOP_HIT | 1.00 | -3.73% |
| SELL | retest2 | 2025-04-09 10:15:00 | 1118.35 | 2025-04-16 11:15:00 | 1215.20 | STOP_HIT | 1.00 | -8.66% |
| BUY | retest2 | 2025-06-20 15:15:00 | 1349.30 | 2025-08-07 10:15:00 | 1332.50 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-06-23 11:15:00 | 1358.90 | 2025-08-07 10:15:00 | 1332.50 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-08-04 09:15:00 | 1362.80 | 2025-08-07 10:15:00 | 1332.50 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-09-30 10:15:00 | 1398.60 | 2026-01-19 15:15:00 | 1403.00 | STOP_HIT | 1.00 | 0.31% |
| BUY | retest2 | 2025-10-08 12:15:00 | 1399.50 | 2026-01-19 15:15:00 | 1403.00 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2026-04-08 14:15:00 | 1452.70 | 2026-04-15 09:15:00 | 1485.60 | STOP_HIT | 1.00 | -2.26% |
