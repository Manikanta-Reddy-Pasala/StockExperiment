# RELIANCE (RELIANCE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:30:00 (4998 bars)
- **Last close:** 1437.90
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 2 |
| ALERT3 | 9 |
| PENDING | 36 |
| PENDING_CANCEL | 9 |
| ENTRY1 | 8 |
| ENTRY2 | 19 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 26 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 27 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 4 / 23
- **Target hits / Stop hits / Partials:** 0 / 26 / 1
- **Avg / median % per leg:** -0.39% / -1.40%
- **Sum % (uncompounded):** -10.53%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 2 | 13.3% | 0 | 15 | 0 | -1.79% | -26.9% |
| BUY @ 2nd Alert (retest1) | 7 | 0 | 0.0% | 0 | 7 | 0 | -2.42% | -17.0% |
| BUY @ 3rd Alert (retest2) | 8 | 2 | 25.0% | 0 | 8 | 0 | -1.24% | -10.0% |
| SELL (all) | 12 | 2 | 16.7% | 0 | 11 | 1 | 1.37% | 16.4% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.29% | -2.3% |
| SELL @ 3rd Alert (retest2) | 11 | 2 | 18.2% | 0 | 10 | 1 | 1.70% | 18.7% |
| retest1 (combined) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.41% | -19.3% |
| retest2 (combined) | 19 | 4 | 21.1% | 0 | 18 | 1 | 0.46% | 8.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-08 15:15:00 | 1227.88 | 1192.19 | 1192.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-11 09:15:00 | 1229.95 | 1192.56 | 1192.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-13 11:15:00 | 1453.57 | 1456.43 | 1405.22 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-03-27 09:15:00 | 1475.05 | 1448.73 | 1414.07 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-27 10:15:00 | 1480.57 | 1449.04 | 1414.40 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-04-08 09:15:00 | 1484.80 | 1459.87 | 1428.02 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-08 10:15:00 | 1482.35 | 1460.09 | 1428.29 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-04-10 09:15:00 | 1486.03 | 1462.30 | 1431.44 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-10 10:15:00 | 1474.78 | 1462.43 | 1431.66 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-04-15 10:15:00 | 1476.15 | 1464.05 | 1434.59 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-15 11:15:00 | 1479.95 | 1464.21 | 1434.81 | BUY ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 11:15:00 | 1438.70 | 1464.77 | 1445.26 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-05-03 11:15:00 | 1445.26 | 1464.77 | 1445.26 | SL hit qty=1.00 sl=1445.26 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-05-03 11:15:00 | 1445.26 | 1464.77 | 1445.26 | SL hit qty=1.00 sl=1445.26 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-05-03 11:15:00 | 1445.26 | 1464.77 | 1445.26 | SL hit qty=1.00 sl=1445.26 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-05-03 11:15:00 | 1445.26 | 1464.77 | 1445.26 | SL hit qty=1.00 sl=1445.26 alert=retest1 |
| Cross detected — sustain check pending | 2024-05-22 09:15:00 | 1459.60 | 1438.05 | 1435.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 10:15:00 | 1458.00 | 1438.25 | 1435.59 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-05-30 09:15:00 | 1432.12 | 1447.16 | 1441.09 | SL hit qty=1.00 sl=1432.12 alert=retest2 |
| Cross detected — sustain check pending | 2024-06-03 09:15:00 | 1477.05 | 1445.19 | 1440.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-03 10:15:00 | 1490.85 | 1445.64 | 1440.72 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-06-04 10:15:00 | 1432.12 | 1448.11 | 1442.16 | SL hit qty=1.00 sl=1432.12 alert=retest2 |
| Cross detected — sustain check pending | 2024-06-07 11:15:00 | 1462.97 | 1443.06 | 1440.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 12:15:00 | 1460.80 | 1443.24 | 1440.16 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-21 13:15:00 | 1453.47 | 1455.46 | 1448.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-21 14:15:00 | 1451.00 | 1455.41 | 1448.13 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-06-21 15:15:00 | 1455.07 | 1455.41 | 1448.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-24 09:15:00 | 1442.05 | 1455.27 | 1448.13 | ENTRY2 sustain failed after 3960m |
| Cross detected — sustain check pending | 2024-06-25 14:15:00 | 1454.70 | 1454.22 | 1448.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-25 15:15:00 | 1452.53 | 1454.20 | 1448.02 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-06-26 09:15:00 | 1467.45 | 1454.33 | 1448.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 10:15:00 | 1477.60 | 1454.57 | 1448.27 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 1504.57 | 1535.91 | 1503.06 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-07-26 15:15:00 | 1509.30 | 1524.88 | 1501.93 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 09:15:00 | 1514.25 | 1524.77 | 1501.99 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 3960m) |
| Stop hit — per-position SL triggered | 2024-07-31 09:15:00 | 1501.70 | 1523.74 | 1503.00 | SL hit qty=1.00 sl=1501.70 alert=retest2 |
| Cross detected — sustain check pending | 2024-08-01 09:15:00 | 1512.00 | 1522.55 | 1503.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 10:15:00 | 1515.18 | 1522.48 | 1503.17 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-08-02 09:15:00 | 1501.70 | 1521.84 | 1503.42 | SL hit qty=1.00 sl=1501.70 alert=retest2 |
| Cross detected — sustain check pending | 2024-08-26 09:15:00 | 1513.93 | 1495.54 | 1493.59 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 10:15:00 | 1516.60 | 1495.75 | 1493.70 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-08-27 10:15:00 | 1501.70 | 1496.85 | 1494.33 | SL hit qty=1.00 sl=1501.70 alert=retest2 |
| Cross detected — sustain check pending | 2024-08-29 13:15:00 | 1531.85 | 1497.84 | 1495.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 14:15:00 | 1518.72 | 1498.05 | 1495.17 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-09-04 09:15:00 | 1501.70 | 1501.27 | 1497.18 | SL hit qty=1.00 sl=1501.70 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 14:15:00 | 1496.00 | 1502.37 | 1498.00 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-09-11 12:15:00 | 1464.88 | 1493.99 | 1494.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-11 12:15:00 | 1464.88 | 1493.99 | 1494.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 12:15:00 | 1464.88 | 1493.99 | 1494.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 13:15:00 | 1453.20 | 1493.59 | 1493.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 10:15:00 | 1487.90 | 1484.61 | 1488.78 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 11:15:00 | 1489.55 | 1484.66 | 1488.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 1489.55 | 1484.66 | 1488.79 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-09-20 13:15:00 | 1484.75 | 1484.73 | 1488.79 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 14:15:00 | 1485.07 | 1484.74 | 1488.77 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-09-20 14:15:00 | 1492.50 | 1484.74 | 1488.77 | SL hit qty=1.00 sl=1492.50 alert=retest2 |
| Cross detected — sustain check pending | 2024-09-25 09:15:00 | 1484.95 | 1485.97 | 1489.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-25 10:15:00 | 1483.72 | 1485.95 | 1489.07 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-09-25 13:15:00 | 1492.50 | 1486.04 | 1489.08 | SL hit qty=1.00 sl=1492.50 alert=retest2 |
| Cross detected — sustain check pending | 2024-09-30 13:15:00 | 1480.78 | 1488.47 | 1490.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 14:15:00 | 1475.07 | 1488.33 | 1489.98 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-11-13 13:15:00 | 1253.81 | 1343.89 | 1389.39 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-24 13:15:00 | 1300.10 | 1249.12 | 1249.00 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2025-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 13:15:00 | 1300.10 | 1249.12 | 1249.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 14:15:00 | 1302.20 | 1249.65 | 1249.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 1475.70 | 1481.25 | 1439.06 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-07-16 11:15:00 | 1483.00 | 1481.25 | 1439.48 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-16 12:15:00 | 1480.70 | 1481.25 | 1439.68 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-07-16 13:15:00 | 1485.00 | 1481.28 | 1439.91 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 14:15:00 | 1485.10 | 1481.32 | 1440.14 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-17 10:15:00 | 1481.80 | 1481.36 | 1440.77 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-17 11:15:00 | 1482.70 | 1481.37 | 1440.98 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1446.90 | 1480.49 | 1442.88 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 1442.88 | 1480.49 | 1442.88 | SL hit qty=1.00 sl=1442.88 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 1442.88 | 1480.49 | 1442.88 | SL hit qty=1.00 sl=1442.88 alert=retest1 |

### Cycle 4 — SELL (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 15:15:00 | 1384.70 | 1423.43 | 1423.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 13:15:00 | 1381.10 | 1421.69 | 1422.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 12:15:00 | 1413.60 | 1413.28 | 1417.97 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 12:15:00 | 1413.60 | 1413.28 | 1417.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 1413.60 | 1413.28 | 1417.97 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-08-20 15:15:00 | 1411.30 | 1413.76 | 1417.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-21 09:15:00 | 1429.80 | 1413.92 | 1418.05 | ENTRY2 sustain failed after 1080m |
| Cross detected — sustain check pending | 2025-08-22 14:15:00 | 1408.90 | 1414.65 | 1418.19 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 15:15:00 | 1410.00 | 1414.60 | 1418.15 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-25 13:15:00 | 1418.30 | 1414.49 | 1418.00 | SL hit qty=1.00 sl=1418.30 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-25 14:15:00 | 1410.20 | 1414.45 | 1417.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-25 15:15:00 | 1412.60 | 1414.43 | 1417.94 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-26 09:15:00 | 1402.20 | 1414.31 | 1417.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 10:15:00 | 1398.70 | 1414.15 | 1417.76 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 1418.30 | 1394.75 | 1403.12 | SL hit qty=1.00 sl=1418.30 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-19 09:15:00 | 1405.80 | 1395.99 | 1403.46 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 10:15:00 | 1405.30 | 1396.09 | 1403.47 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-19 14:15:00 | 1406.30 | 1396.63 | 1403.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 15:15:00 | 1409.40 | 1396.75 | 1403.63 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 1409.00 | 1396.88 | 1403.65 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-09-22 14:15:00 | 1390.20 | 1397.12 | 1403.61 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 15:15:00 | 1390.90 | 1397.06 | 1403.55 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-17 09:15:00 | 1410.30 | 1383.20 | 1391.57 | SL hit qty=1.00 sl=1410.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 11:15:00 | 1418.30 | 1383.90 | 1391.84 | SL hit qty=1.00 sl=1418.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 11:15:00 | 1418.30 | 1383.90 | 1391.84 | SL hit qty=1.00 sl=1418.30 alert=retest2 |

### Cycle 5 — BUY (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 12:15:00 | 1459.00 | 1399.06 | 1398.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 09:15:00 | 1482.40 | 1401.41 | 1400.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-30 14:15:00 | 1539.30 | 1541.16 | 1511.64 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-12-31 09:15:00 | 1548.00 | 1541.23 | 1511.96 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 10:15:00 | 1555.00 | 1541.36 | 1512.18 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 1526.20 | 1550.52 | 1520.75 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 1520.75 | 1550.52 | 1520.75 | SL hit qty=1.00 sl=1520.75 alert=retest1 |

### Cycle 6 — SELL (started 2026-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 10:15:00 | 1384.00 | 1501.19 | 1501.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 09:15:00 | 1363.30 | 1460.02 | 1478.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 09:15:00 | 1459.10 | 1454.50 | 1474.41 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-02-05 13:15:00 | 1441.00 | 1454.07 | 1473.12 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-05 14:15:00 | 1444.50 | 1453.97 | 1472.98 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-05 15:15:00 | 1441.90 | 1453.85 | 1472.83 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-06 09:15:00 | 1437.90 | 1453.69 | 1472.65 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 1080m) |
| Cross detected — sustain check pending | 2026-02-06 12:15:00 | 1440.50 | 1453.40 | 1472.22 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-06 13:15:00 | 1442.00 | 1453.29 | 1472.07 | ENTRY1 sustain failed after 60m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 1464.40 | 1455.06 | 1470.87 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 1470.87 | 1455.06 | 1470.87 | SL hit qty=1.00 sl=1470.87 alert=retest1 |
| Cross detected — sustain check pending | 2026-02-12 11:15:00 | 1457.20 | 1455.15 | 1470.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 12:15:00 | 1454.70 | 1455.14 | 1470.68 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-05-05 09:15:00 | 1473.00 | 1377.61 | 1392.85 | SL hit qty=1.00 sl=1473.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-05-05 10:15:00 | 1452.30 | 1378.35 | 1393.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 11:15:00 | 1460.10 | 1379.16 | 1393.48 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-05-05 13:15:00 | 1460.40 | 1380.83 | 1394.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-05-05 14:15:00 | 1464.40 | 1381.66 | 1394.53 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-05-06 09:15:00 | 1455.50 | 1383.19 | 1395.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Stop hit — per-position SL triggered | 2026-05-06 09:15:00 | 1473.00 | 1383.19 | 1395.17 | SL hit qty=1.00 sl=1473.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 10:15:00 | 1447.30 | 1383.83 | 1395.43 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-03-27 10:15:00 | 1480.57 | 2024-05-03 11:15:00 | 1445.26 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest1 | 2024-04-08 10:15:00 | 1482.35 | 2024-05-03 11:15:00 | 1445.26 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest1 | 2024-04-10 10:15:00 | 1474.78 | 2024-05-03 11:15:00 | 1445.26 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest1 | 2024-04-15 11:15:00 | 1479.95 | 2024-05-03 11:15:00 | 1445.26 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2024-05-22 10:15:00 | 1458.00 | 2024-05-30 09:15:00 | 1432.12 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2024-06-03 10:15:00 | 1490.85 | 2024-06-04 10:15:00 | 1432.12 | STOP_HIT | 1.00 | -3.94% |
| BUY | retest2 | 2024-06-07 12:15:00 | 1460.80 | 2024-07-31 09:15:00 | 1501.70 | STOP_HIT | 1.00 | 2.80% |
| BUY | retest2 | 2024-06-26 10:15:00 | 1477.60 | 2024-08-02 09:15:00 | 1501.70 | STOP_HIT | 1.00 | 1.63% |
| BUY | retest2 | 2024-07-29 09:15:00 | 1514.25 | 2024-08-27 10:15:00 | 1501.70 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2024-08-01 10:15:00 | 1515.18 | 2024-09-04 09:15:00 | 1501.70 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-08-26 10:15:00 | 1516.60 | 2024-09-11 12:15:00 | 1464.88 | STOP_HIT | 1.00 | -3.41% |
| BUY | retest2 | 2024-08-29 14:15:00 | 1518.72 | 2024-09-11 12:15:00 | 1464.88 | STOP_HIT | 1.00 | -3.55% |
| SELL | retest2 | 2024-09-20 14:15:00 | 1485.07 | 2024-09-20 14:15:00 | 1492.50 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2024-09-25 10:15:00 | 1483.72 | 2024-09-25 13:15:00 | 1492.50 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2024-09-30 14:15:00 | 1475.07 | 2024-11-13 13:15:00 | 1253.81 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2024-09-30 14:15:00 | 1475.07 | 2025-04-24 13:15:00 | 1300.10 | STOP_HIT | 0.50 | 11.86% |
| BUY | retest1 | 2025-07-16 14:15:00 | 1485.10 | 2025-07-21 09:15:00 | 1442.88 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest1 | 2025-07-17 11:15:00 | 1482.70 | 2025-07-21 09:15:00 | 1442.88 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2025-08-22 15:15:00 | 1410.00 | 2025-08-25 13:15:00 | 1418.30 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-08-26 10:15:00 | 1398.70 | 2025-09-18 09:15:00 | 1418.30 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-09-19 10:15:00 | 1405.30 | 2025-10-17 09:15:00 | 1410.30 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-09-19 15:15:00 | 1409.40 | 2025-10-17 11:15:00 | 1418.30 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-09-22 15:15:00 | 1390.90 | 2025-10-17 11:15:00 | 1418.30 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest1 | 2025-12-31 10:15:00 | 1555.00 | 2026-01-06 09:15:00 | 1520.75 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest1 | 2026-02-06 09:15:00 | 1437.90 | 2026-02-12 09:15:00 | 1470.87 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2026-02-12 12:15:00 | 1454.70 | 2026-05-05 09:15:00 | 1473.00 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2026-05-05 11:15:00 | 1460.10 | 2026-05-06 09:15:00 | 1473.00 | STOP_HIT | 1.00 | -0.88% |
