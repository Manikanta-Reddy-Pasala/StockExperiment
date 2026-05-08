# RELIANCE (RELIANCE)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:30:00 (4998 bars)
- **Last close:** 1435.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty booked @ 5% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 2 |
| ALERT3 | 10 |
| PENDING | 39 |
| PENDING_CANCEL | 9 |
| ENTRY1 | 8 |
| ENTRY2 | 22 |
| PARTIAL | 1 |
| TARGET_HIT | 3 |
| STOP_HIT | 25 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 29 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 8 / 21
- **Target hits / Stop hits / Partials:** 3 / 25 / 1
- **Avg / median % per leg:** -0.35% / -1.36%
- **Sum % (uncompounded):** -10.03%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 2 | 13.3% | 1 | 14 | 0 | -1.54% | -23.2% |
| BUY @ 2nd Alert (retest1) | 7 | 0 | 0.0% | 0 | 7 | 0 | -2.86% | -20.0% |
| BUY @ 3rd Alert (retest2) | 8 | 2 | 25.0% | 1 | 7 | 0 | -0.39% | -3.1% |
| SELL (all) | 14 | 6 | 42.9% | 2 | 11 | 1 | 0.94% | 13.1% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 1 | 0 | 1 | 6.97% | 13.9% |
| SELL @ 3rd Alert (retest2) | 12 | 4 | 33.3% | 1 | 11 | 0 | -0.07% | -0.8% |
| retest1 (combined) | 9 | 2 | 22.2% | 1 | 7 | 1 | -0.68% | -6.1% |
| retest2 (combined) | 20 | 6 | 30.0% | 2 | 18 | 0 | -0.20% | -3.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 15:15:00 | 1230.00 | 1194.75 | 1194.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 09:15:00 | 1232.95 | 1197.93 | 1196.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-13 11:15:00 | 1453.57 | 1456.43 | 1405.37 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-03-27 09:15:00 | 1475.05 | 1448.73 | 1414.18 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-27 10:15:00 | 1480.57 | 1449.04 | 1414.51 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-04-08 09:15:00 | 1484.80 | 1459.87 | 1428.10 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-08 10:15:00 | 1482.35 | 1460.09 | 1428.37 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-04-10 09:15:00 | 1486.03 | 1462.30 | 1431.52 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-10 10:15:00 | 1474.78 | 1462.43 | 1431.74 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-04-15 10:15:00 | 1476.15 | 1464.05 | 1434.66 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-15 11:15:00 | 1479.95 | 1464.21 | 1434.89 | BUY ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 11:15:00 | 1438.70 | 1464.77 | 1445.31 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-05-03 11:15:00 | 1438.70 | 1464.77 | 1445.31 | SL hit (close<ema400) qty=1.00 sl=1445.31 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-05-03 11:15:00 | 1438.70 | 1464.77 | 1445.31 | SL hit (close<ema400) qty=1.00 sl=1445.31 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-05-03 11:15:00 | 1438.70 | 1464.77 | 1445.31 | SL hit (close<ema400) qty=1.00 sl=1445.31 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-05-03 11:15:00 | 1438.70 | 1464.77 | 1445.31 | SL hit (close<ema400) qty=1.00 sl=1445.31 alert=retest1 |
| Cross detected — sustain check pending | 2024-05-22 09:15:00 | 1459.60 | 1438.05 | 1435.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 10:15:00 | 1458.00 | 1438.25 | 1435.62 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-05-30 09:15:00 | 1426.15 | 1447.16 | 1441.12 | SL hit (close<static) qty=1.00 sl=1432.12 alert=retest2 |
| Cross detected — sustain check pending | 2024-06-03 09:15:00 | 1477.05 | 1445.19 | 1440.50 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-03 10:15:00 | 1490.85 | 1445.64 | 1440.75 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-06-04 10:15:00 | 1403.60 | 1448.11 | 1442.18 | SL hit (close<static) qty=1.00 sl=1432.12 alert=retest2 |
| Cross detected — sustain check pending | 2024-06-07 11:15:00 | 1462.97 | 1443.06 | 1440.08 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 12:15:00 | 1460.80 | 1443.24 | 1440.18 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-21 13:15:00 | 1453.47 | 1455.46 | 1448.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-21 14:15:00 | 1451.00 | 1455.41 | 1448.14 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-06-21 15:15:00 | 1455.07 | 1455.41 | 1448.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-24 09:15:00 | 1442.05 | 1455.27 | 1448.15 | ENTRY2 sustain failed after 3960m |
| Cross detected — sustain check pending | 2024-06-25 14:15:00 | 1454.70 | 1454.22 | 1448.02 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-25 15:15:00 | 1452.53 | 1454.20 | 1448.04 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-06-26 09:15:00 | 1467.45 | 1454.33 | 1448.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 10:15:00 | 1477.60 | 1454.57 | 1448.28 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Target hit | 2024-07-08 12:15:00 | 1606.88 | 1500.29 | 1475.64 | Target hit (10%) qty=1.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 1504.57 | 1535.91 | 1503.06 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-07-26 15:15:00 | 1509.30 | 1524.88 | 1501.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 09:15:00 | 1514.25 | 1524.77 | 1502.00 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 3960m) |
| Cross detected — sustain check pending | 2024-08-01 09:15:00 | 1512.00 | 1522.55 | 1503.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 10:15:00 | 1515.18 | 1522.48 | 1503.18 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-08-02 11:15:00 | 1501.00 | 1521.44 | 1503.41 | SL hit (close<static) qty=1.00 sl=1501.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-02 11:15:00 | 1501.00 | 1521.44 | 1503.41 | SL hit (close<static) qty=1.00 sl=1501.70 alert=retest2 |
| Cross detected — sustain check pending | 2024-08-26 09:15:00 | 1513.93 | 1495.54 | 1493.59 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 10:15:00 | 1516.60 | 1495.75 | 1493.71 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-08-27 12:15:00 | 1501.68 | 1496.95 | 1494.41 | SL hit (close<static) qty=1.00 sl=1501.70 alert=retest2 |
| Cross detected — sustain check pending | 2024-08-29 13:15:00 | 1531.85 | 1497.84 | 1495.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 14:15:00 | 1518.72 | 1498.05 | 1495.17 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 14:15:00 | 1496.00 | 1502.37 | 1498.00 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-05 14:15:00 | 1496.00 | 1502.37 | 1498.00 | SL hit (close<static) qty=1.00 sl=1501.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-11 12:15:00 | 1464.88 | 1493.99 | 1494.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 12:15:00 | 1464.88 | 1493.99 | 1494.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 13:15:00 | 1453.20 | 1493.59 | 1493.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 10:15:00 | 1487.90 | 1484.61 | 1488.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 11:15:00 | 1489.55 | 1484.66 | 1488.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 1489.55 | 1484.66 | 1488.79 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-09-20 13:15:00 | 1484.75 | 1484.73 | 1488.79 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 14:15:00 | 1485.07 | 1484.74 | 1488.77 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-09-23 09:15:00 | 1497.82 | 1484.87 | 1488.80 | SL hit (close>static) qty=1.00 sl=1492.50 alert=retest2 |
| Cross detected — sustain check pending | 2024-09-25 09:15:00 | 1484.95 | 1485.97 | 1489.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-25 10:15:00 | 1483.72 | 1485.95 | 1489.07 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-09-25 13:15:00 | 1495.97 | 1486.04 | 1489.08 | SL hit (close>static) qty=1.00 sl=1492.50 alert=retest2 |
| Cross detected — sustain check pending | 2024-09-30 13:15:00 | 1480.78 | 1488.47 | 1490.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 14:15:00 | 1475.07 | 1488.33 | 1489.98 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Target hit | 2024-10-24 10:15:00 | 1327.57 | 1407.42 | 1439.19 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 13:15:00 | 1300.10 | 1249.12 | 1249.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 14:15:00 | 1302.20 | 1249.65 | 1249.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 1475.70 | 1481.25 | 1439.06 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-07-16 11:15:00 | 1483.00 | 1481.25 | 1439.48 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-16 12:15:00 | 1480.70 | 1481.25 | 1439.68 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-07-16 13:15:00 | 1485.00 | 1481.28 | 1439.91 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 14:15:00 | 1485.10 | 1481.32 | 1440.14 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-17 10:15:00 | 1481.80 | 1481.36 | 1440.77 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-17 11:15:00 | 1482.70 | 1481.37 | 1440.98 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1446.90 | 1480.49 | 1442.88 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-21 11:15:00 | 1437.70 | 1479.72 | 1442.87 | SL hit (close<ema400) qty=1.00 sl=1442.87 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-21 11:15:00 | 1437.70 | 1479.72 | 1442.87 | SL hit (close<ema400) qty=1.00 sl=1442.87 alert=retest1 |

### Cycle 4 — SELL (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 15:15:00 | 1384.70 | 1423.43 | 1423.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 13:15:00 | 1381.10 | 1421.69 | 1422.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 12:15:00 | 1413.60 | 1413.28 | 1417.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 12:15:00 | 1413.60 | 1413.28 | 1417.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 1413.60 | 1413.28 | 1417.97 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-08-20 15:15:00 | 1411.30 | 1413.76 | 1417.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-21 09:15:00 | 1429.80 | 1413.92 | 1418.05 | ENTRY2 sustain failed after 1080m |
| Cross detected — sustain check pending | 2025-08-22 14:15:00 | 1408.90 | 1414.65 | 1418.19 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 15:15:00 | 1410.00 | 1414.60 | 1418.15 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-25 14:15:00 | 1410.20 | 1414.45 | 1417.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-25 15:15:00 | 1412.60 | 1414.43 | 1417.94 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-26 09:15:00 | 1402.20 | 1414.31 | 1417.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 10:15:00 | 1398.70 | 1414.15 | 1417.76 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-19 09:15:00 | 1405.80 | 1395.99 | 1403.46 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 10:15:00 | 1405.30 | 1396.09 | 1403.47 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-19 14:15:00 | 1406.30 | 1396.63 | 1403.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 15:15:00 | 1409.40 | 1396.75 | 1403.63 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 1390.20 | 1397.12 | 1403.61 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-09-23 09:15:00 | 1383.50 | 1396.92 | 1403.45 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 10:15:00 | 1382.90 | 1396.79 | 1403.35 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-24 09:15:00 | 1386.10 | 1396.26 | 1402.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 10:15:00 | 1387.30 | 1396.17 | 1402.81 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-24 14:15:00 | 1382.10 | 1395.91 | 1402.55 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 15:15:00 | 1383.00 | 1395.79 | 1402.45 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-07 14:15:00 | 1383.20 | 1386.17 | 1395.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 15:15:00 | 1384.40 | 1386.16 | 1395.33 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 1397.30 | 1382.80 | 1391.46 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-17 09:15:00 | 1407.80 | 1383.20 | 1391.57 | SL hit (close>static) qty=1.00 sl=1404.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 09:15:00 | 1407.80 | 1383.20 | 1391.57 | SL hit (close>static) qty=1.00 sl=1404.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 09:15:00 | 1407.80 | 1383.20 | 1391.57 | SL hit (close>static) qty=1.00 sl=1404.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 09:15:00 | 1407.80 | 1383.20 | 1391.57 | SL hit (close>static) qty=1.00 sl=1404.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 11:15:00 | 1421.10 | 1383.90 | 1391.84 | SL hit (close>static) qty=1.00 sl=1418.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 11:15:00 | 1421.10 | 1383.90 | 1391.84 | SL hit (close>static) qty=1.00 sl=1418.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 11:15:00 | 1421.10 | 1383.90 | 1391.84 | SL hit (close>static) qty=1.00 sl=1418.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 11:15:00 | 1421.10 | 1383.90 | 1391.84 | SL hit (close>static) qty=1.00 sl=1418.30 alert=retest2 |

### Cycle 5 — BUY (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 12:15:00 | 1459.00 | 1399.06 | 1398.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 09:15:00 | 1482.40 | 1401.41 | 1400.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-30 14:15:00 | 1539.30 | 1541.16 | 1511.64 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-12-31 09:15:00 | 1548.00 | 1541.23 | 1511.96 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 10:15:00 | 1555.00 | 1541.36 | 1512.18 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 1526.20 | 1550.52 | 1520.75 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-06 10:15:00 | 1511.20 | 1550.13 | 1520.70 | SL hit (close<ema400) qty=1.00 sl=1520.70 alert=retest1 |

### Cycle 6 — SELL (started 2026-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 10:15:00 | 1384.00 | 1501.19 | 1501.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 09:15:00 | 1363.30 | 1460.02 | 1478.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 09:15:00 | 1459.10 | 1454.50 | 1474.41 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-02-05 13:15:00 | 1441.00 | 1454.07 | 1473.12 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-05 14:15:00 | 1444.50 | 1453.97 | 1472.98 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-05 15:15:00 | 1441.90 | 1453.85 | 1472.83 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-06 09:15:00 | 1437.90 | 1453.69 | 1472.65 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 1080m) |
| Cross detected — sustain check pending | 2026-02-06 12:15:00 | 1440.50 | 1453.40 | 1472.22 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-06 13:15:00 | 1442.00 | 1453.29 | 1472.07 | ENTRY1 sustain failed after 60m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 1464.40 | 1455.06 | 1470.87 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-02-12 11:15:00 | 1457.20 | 1455.15 | 1470.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 12:15:00 | 1454.70 | 1455.14 | 1470.68 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1366.01 | 1434.32 | 1453.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2026-03-04 09:15:00 | 1309.23 | 1428.91 | 1449.98 | Target hit (10%) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 14:15:00 | 1425.00 | 1419.83 | 1442.47 | SL hit (close>ema200) qty=0.50 sl=1419.83 alert=retest1 |
| Cross detected — sustain check pending | 2026-05-05 10:15:00 | 1452.30 | 1378.35 | 1393.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 11:15:00 | 1460.10 | 1379.16 | 1393.48 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-05-05 13:15:00 | 1460.40 | 1380.83 | 1394.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-05-05 14:15:00 | 1464.40 | 1381.66 | 1394.53 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-05-06 09:15:00 | 1455.50 | 1383.19 | 1395.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 10:15:00 | 1447.30 | 1383.83 | 1395.43 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-03-27 10:15:00 | 1480.57 | 2024-05-03 11:15:00 | 1438.70 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest1 | 2024-04-08 10:15:00 | 1482.35 | 2024-05-03 11:15:00 | 1438.70 | STOP_HIT | 1.00 | -2.94% |
| BUY | retest1 | 2024-04-10 10:15:00 | 1474.78 | 2024-05-03 11:15:00 | 1438.70 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest1 | 2024-04-15 11:15:00 | 1479.95 | 2024-05-03 11:15:00 | 1438.70 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2024-05-22 10:15:00 | 1458.00 | 2024-05-30 09:15:00 | 1426.15 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2024-06-03 10:15:00 | 1490.85 | 2024-06-04 10:15:00 | 1403.60 | STOP_HIT | 1.00 | -5.85% |
| BUY | retest2 | 2024-06-07 12:15:00 | 1460.80 | 2024-07-08 12:15:00 | 1606.88 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-26 10:15:00 | 1477.60 | 2024-08-02 11:15:00 | 1501.00 | STOP_HIT | 1.00 | 1.58% |
| BUY | retest2 | 2024-07-29 09:15:00 | 1514.25 | 2024-08-02 11:15:00 | 1501.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-08-01 10:15:00 | 1515.18 | 2024-08-27 12:15:00 | 1501.68 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-08-26 10:15:00 | 1516.60 | 2024-09-05 14:15:00 | 1496.00 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2024-08-29 14:15:00 | 1518.72 | 2024-09-11 12:15:00 | 1464.88 | STOP_HIT | 1.00 | -3.55% |
| SELL | retest2 | 2024-09-20 14:15:00 | 1485.07 | 2024-09-23 09:15:00 | 1497.82 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-09-25 10:15:00 | 1483.72 | 2024-09-25 13:15:00 | 1495.97 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-09-30 14:15:00 | 1475.07 | 2024-10-24 10:15:00 | 1327.57 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest1 | 2025-07-16 14:15:00 | 1485.10 | 2025-07-21 11:15:00 | 1437.70 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest1 | 2025-07-17 11:15:00 | 1482.70 | 2025-07-21 11:15:00 | 1437.70 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2025-08-22 15:15:00 | 1410.00 | 2025-10-17 09:15:00 | 1407.80 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2025-08-26 10:15:00 | 1398.70 | 2025-10-17 09:15:00 | 1407.80 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-09-19 10:15:00 | 1405.30 | 2025-10-17 09:15:00 | 1407.80 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2025-09-19 15:15:00 | 1409.40 | 2025-10-17 09:15:00 | 1407.80 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2025-09-23 10:15:00 | 1382.90 | 2025-10-17 11:15:00 | 1421.10 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2025-09-24 10:15:00 | 1387.30 | 2025-10-17 11:15:00 | 1421.10 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-09-24 15:15:00 | 1383.00 | 2025-10-17 11:15:00 | 1421.10 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2025-10-07 15:15:00 | 1384.40 | 2025-10-17 11:15:00 | 1421.10 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest1 | 2025-12-31 10:15:00 | 1555.00 | 2026-01-06 10:15:00 | 1511.20 | STOP_HIT | 1.00 | -2.82% |
| SELL | retest1 | 2026-02-06 09:15:00 | 1437.90 | 2026-03-02 09:15:00 | 1366.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-02-06 09:15:00 | 1437.90 | 2026-03-04 09:15:00 | 1309.23 | TARGET_HIT | 0.50 | 8.95% |
| SELL | retest2 | 2026-02-12 12:15:00 | 1454.70 | 2026-03-09 14:15:00 | 1425.00 | STOP_HIT | 1.00 | 2.04% |
