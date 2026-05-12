# Ipca Laboratories Ltd. (IPCALAB)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1554.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 2 |
| ALERT3 | 75 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 85 |
| PARTIAL | 10 |
| TARGET_HIT | 22 |
| STOP_HIT | 57 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 89 (incl. partial bookings)
- **Trades open at end:** 6
- **Winners / losers:** 44 / 45
- **Target hits / Stop hits / Partials:** 22 / 57 / 10
- **Avg / median % per leg:** 1.68% / -0.86%
- **Sum % (uncompounded):** 149.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 47 | 23 | 48.9% | 21 | 26 | 0 | 3.46% | 162.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 47 | 23 | 48.9% | 21 | 26 | 0 | 3.46% | 162.7% |
| SELL (all) | 42 | 21 | 50.0% | 1 | 31 | 10 | -0.32% | -13.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 42 | 21 | 50.0% | 1 | 31 | 10 | -0.32% | -13.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 89 | 44 | 49.4% | 22 | 57 | 10 | 1.68% | 149.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-06 11:15:00 | 755.65 | 734.00 | 733.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-06 13:15:00 | 758.90 | 734.46 | 734.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-01 10:15:00 | 866.40 | 869.26 | 834.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-01 11:00:00 | 866.40 | 869.26 | 834.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 09:15:00 | 1072.90 | 1105.61 | 1072.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-18 09:45:00 | 1064.30 | 1105.61 | 1072.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 10:15:00 | 1083.00 | 1105.38 | 1072.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-18 11:30:00 | 1086.75 | 1105.24 | 1072.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-18 14:30:00 | 1086.60 | 1104.56 | 1073.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-19 09:15:00 | 1089.05 | 1104.27 | 1073.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-19 10:00:00 | 1086.40 | 1104.10 | 1073.14 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 15:15:00 | 1072.00 | 1102.24 | 1074.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 09:15:00 | 1069.55 | 1102.24 | 1074.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 1068.80 | 1101.91 | 1074.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 10:00:00 | 1068.80 | 1101.91 | 1074.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 10:15:00 | 1082.55 | 1101.71 | 1074.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-23 11:15:00 | 1086.65 | 1101.71 | 1074.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-23 13:00:00 | 1084.90 | 1101.37 | 1074.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-23 15:00:00 | 1086.40 | 1101.09 | 1074.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-24 09:30:00 | 1086.25 | 1100.66 | 1074.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-02-06 09:15:00 | 1193.39 | 1107.61 | 1084.63 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-06-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-07 14:15:00 | 1169.90 | 1250.21 | 1250.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-18 11:15:00 | 1164.15 | 1228.60 | 1238.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-04 10:15:00 | 1180.15 | 1173.23 | 1201.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-04 11:00:00 | 1180.15 | 1173.23 | 1201.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 12:15:00 | 1209.05 | 1176.47 | 1200.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 13:00:00 | 1209.05 | 1176.47 | 1200.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 13:15:00 | 1219.65 | 1176.90 | 1200.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 13:45:00 | 1224.00 | 1176.90 | 1200.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 11:15:00 | 1215.90 | 1178.86 | 1200.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 12:00:00 | 1215.90 | 1178.86 | 1200.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 13:15:00 | 1223.80 | 1184.81 | 1202.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 14:00:00 | 1223.80 | 1184.81 | 1202.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 12:15:00 | 1211.85 | 1193.77 | 1204.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 13:15:00 | 1208.35 | 1193.77 | 1204.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 1208.45 | 1194.01 | 1204.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:00:00 | 1208.45 | 1194.01 | 1204.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 1207.55 | 1194.15 | 1204.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:30:00 | 1211.55 | 1194.15 | 1204.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 1210.55 | 1194.47 | 1204.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:45:00 | 1211.70 | 1194.47 | 1204.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 1231.20 | 1196.46 | 1205.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 14:00:00 | 1231.20 | 1196.46 | 1205.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2024-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 13:15:00 | 1295.90 | 1213.27 | 1213.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-30 13:15:00 | 1299.65 | 1218.82 | 1215.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 1560.00 | 1564.30 | 1503.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 10:00:00 | 1560.00 | 1564.30 | 1503.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 10:15:00 | 1514.90 | 1562.41 | 1510.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 11:00:00 | 1514.90 | 1562.41 | 1510.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 11:15:00 | 1509.40 | 1561.88 | 1510.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-13 13:00:00 | 1520.00 | 1561.46 | 1510.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-14 10:30:00 | 1518.95 | 1558.97 | 1510.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-14 11:15:00 | 1520.00 | 1558.97 | 1510.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-18 10:00:00 | 1534.15 | 1557.59 | 1511.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 12:15:00 | 1524.50 | 1565.15 | 1524.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 13:00:00 | 1524.50 | 1565.15 | 1524.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 13:15:00 | 1522.85 | 1564.73 | 1524.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 13:30:00 | 1526.35 | 1564.73 | 1524.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 14:15:00 | 1528.45 | 1564.36 | 1524.82 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-28 12:15:00 | 1505.90 | 1562.04 | 1524.63 | SL hit (close<static) qty=1.00 sl=1506.05 alert=retest2 |

### Cycle 4 — SELL (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-29 12:15:00 | 1414.65 | 1564.24 | 1564.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-12 09:15:00 | 1393.00 | 1506.64 | 1530.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 13:15:00 | 1515.30 | 1500.14 | 1525.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-13 14:00:00 | 1515.30 | 1500.14 | 1525.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 1416.75 | 1392.59 | 1439.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 11:15:00 | 1410.10 | 1395.98 | 1439.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 15:15:00 | 1460.00 | 1397.34 | 1439.17 | SL hit (close>static) qty=1.00 sl=1446.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 12:15:00 | 1495.50 | 1419.06 | 1419.05 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 10:15:00 | 1376.50 | 1419.35 | 1419.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-09 15:15:00 | 1368.80 | 1412.30 | 1415.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 09:15:00 | 1416.60 | 1407.52 | 1413.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 1416.60 | 1407.52 | 1413.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 1416.60 | 1407.52 | 1413.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:45:00 | 1415.00 | 1407.52 | 1413.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 1407.90 | 1407.52 | 1413.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 11:30:00 | 1401.90 | 1407.48 | 1413.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 12:45:00 | 1403.30 | 1407.44 | 1412.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 1331.81 | 1393.86 | 1404.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 1333.13 | 1393.86 | 1404.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-30 13:15:00 | 1379.30 | 1375.44 | 1391.96 | SL hit (close>ema200) qty=0.50 sl=1375.44 alert=retest2 |

### Cycle 7 — BUY (started 2025-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 10:15:00 | 1444.70 | 1402.50 | 1402.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 11:15:00 | 1450.60 | 1402.98 | 1402.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 09:15:00 | 1444.20 | 1454.87 | 1434.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-01 09:45:00 | 1442.00 | 1454.87 | 1434.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 1434.20 | 1454.66 | 1434.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:45:00 | 1423.70 | 1454.66 | 1434.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 1432.10 | 1454.44 | 1434.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 12:15:00 | 1444.50 | 1454.21 | 1434.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 13:00:00 | 1440.20 | 1454.07 | 1434.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 14:45:00 | 1442.10 | 1453.85 | 1434.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-05 12:15:00 | 1416.00 | 1452.65 | 1434.59 | SL hit (close<static) qty=1.00 sl=1427.20 alert=retest2 |

### Cycle 8 — SELL (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 09:15:00 | 1371.70 | 1420.89 | 1421.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 12:15:00 | 1357.20 | 1411.30 | 1415.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 12:15:00 | 1417.80 | 1408.85 | 1414.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 12:15:00 | 1417.80 | 1408.85 | 1414.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 1417.80 | 1408.85 | 1414.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 13:00:00 | 1417.80 | 1408.85 | 1414.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 13:15:00 | 1403.60 | 1408.80 | 1414.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 11:30:00 | 1392.30 | 1408.62 | 1414.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 1380.70 | 1408.21 | 1413.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 14:15:00 | 1435.00 | 1407.12 | 1413.24 | SL hit (close>static) qty=1.00 sl=1418.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 13:15:00 | 1441.10 | 1348.96 | 1348.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-21 10:15:00 | 1448.00 | 1352.50 | 1350.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 12:15:00 | 1399.00 | 1414.53 | 1391.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-17 13:00:00 | 1399.00 | 1414.53 | 1391.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 13:15:00 | 1391.70 | 1414.30 | 1391.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 13:45:00 | 1392.20 | 1414.30 | 1391.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 1393.70 | 1414.10 | 1391.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 15:00:00 | 1399.70 | 1412.47 | 1391.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 13:30:00 | 1406.10 | 1412.35 | 1391.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 13:15:00 | 1400.00 | 1415.34 | 1396.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 10:15:00 | 1413.40 | 1414.21 | 1397.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 1424.60 | 1414.31 | 1397.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 11:30:00 | 1429.50 | 1414.55 | 1397.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 10:00:00 | 1425.70 | 1415.42 | 1398.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 10:45:00 | 1425.70 | 1415.53 | 1398.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:15:00 | 1432.50 | 1414.29 | 1399.39 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-08 12:15:00 | 1539.67 | 1426.73 | 1407.10 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-01-18 11:30:00 | 1086.75 | 2024-02-06 09:15:00 | 1193.39 | TARGET_HIT | 1.00 | 9.81% |
| BUY | retest2 | 2024-01-18 14:30:00 | 1086.60 | 2024-02-06 10:15:00 | 1195.43 | TARGET_HIT | 1.00 | 10.02% |
| BUY | retest2 | 2024-01-19 09:15:00 | 1089.05 | 2024-02-06 10:15:00 | 1195.26 | TARGET_HIT | 1.00 | 9.75% |
| BUY | retest2 | 2024-01-19 10:00:00 | 1086.40 | 2024-02-06 10:15:00 | 1197.96 | TARGET_HIT | 1.00 | 10.27% |
| BUY | retest2 | 2024-01-23 11:15:00 | 1086.65 | 2024-02-06 10:15:00 | 1195.04 | TARGET_HIT | 1.00 | 9.97% |
| BUY | retest2 | 2024-01-23 13:00:00 | 1084.90 | 2024-02-06 10:15:00 | 1195.32 | TARGET_HIT | 1.00 | 10.18% |
| BUY | retest2 | 2024-01-23 15:00:00 | 1086.40 | 2024-02-06 10:15:00 | 1195.04 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-24 09:30:00 | 1086.25 | 2024-02-06 10:15:00 | 1194.88 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-15 13:30:00 | 1168.75 | 2024-03-19 09:15:00 | 1143.05 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2024-03-15 14:45:00 | 1169.45 | 2024-03-19 09:15:00 | 1143.05 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2024-03-18 09:15:00 | 1179.95 | 2024-03-19 09:15:00 | 1143.05 | STOP_HIT | 1.00 | -3.13% |
| BUY | retest2 | 2024-03-18 12:15:00 | 1171.50 | 2024-03-19 09:15:00 | 1143.05 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2024-03-20 14:45:00 | 1159.50 | 2024-04-04 13:15:00 | 1275.45 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-21 10:00:00 | 1160.15 | 2024-04-04 13:15:00 | 1276.17 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-05 09:45:00 | 1165.70 | 2024-06-07 14:15:00 | 1169.90 | STOP_HIT | 1.00 | 0.36% |
| BUY | retest2 | 2024-06-06 14:45:00 | 1158.35 | 2024-06-07 14:15:00 | 1169.90 | STOP_HIT | 1.00 | 1.00% |
| BUY | retest2 | 2024-11-13 13:00:00 | 1520.00 | 2024-11-28 12:15:00 | 1505.90 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2024-11-14 10:30:00 | 1518.95 | 2024-11-28 12:15:00 | 1505.90 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2024-11-14 11:15:00 | 1520.00 | 2024-11-28 12:15:00 | 1505.90 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2024-11-18 10:00:00 | 1534.15 | 2024-11-28 12:15:00 | 1505.90 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2024-11-29 10:45:00 | 1532.70 | 2024-12-03 10:15:00 | 1518.45 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2024-12-06 12:30:00 | 1533.90 | 2024-12-09 09:15:00 | 1506.70 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2024-12-10 09:15:00 | 1544.90 | 2024-12-18 13:15:00 | 1514.50 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2024-12-10 15:00:00 | 1533.55 | 2024-12-18 13:15:00 | 1514.50 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-03-25 11:15:00 | 1410.10 | 2025-03-25 15:15:00 | 1460.00 | STOP_HIT | 1.00 | -3.54% |
| SELL | retest2 | 2025-03-27 10:00:00 | 1410.40 | 2025-03-27 14:15:00 | 1490.50 | STOP_HIT | 1.00 | -5.68% |
| SELL | retest2 | 2025-03-27 10:30:00 | 1408.05 | 2025-03-27 14:15:00 | 1490.50 | STOP_HIT | 1.00 | -5.86% |
| SELL | retest2 | 2025-03-27 12:45:00 | 1409.70 | 2025-03-27 14:15:00 | 1490.50 | STOP_HIT | 1.00 | -5.73% |
| SELL | retest2 | 2025-03-28 09:30:00 | 1424.45 | 2025-03-28 14:15:00 | 1503.10 | STOP_HIT | 1.00 | -5.52% |
| SELL | retest2 | 2025-04-01 11:30:00 | 1419.90 | 2025-04-03 14:15:00 | 1490.95 | STOP_HIT | 1.00 | -5.00% |
| SELL | retest2 | 2025-04-04 09:45:00 | 1407.15 | 2025-04-07 09:15:00 | 1266.44 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-21 11:15:00 | 1424.00 | 2025-05-07 09:15:00 | 1352.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-25 10:15:00 | 1407.00 | 2025-05-08 14:15:00 | 1338.17 | PARTIAL | 0.50 | 4.89% |
| SELL | retest2 | 2025-04-25 10:45:00 | 1405.30 | 2025-05-09 09:15:00 | 1336.65 | PARTIAL | 0.50 | 4.89% |
| SELL | retest2 | 2025-04-28 09:15:00 | 1406.90 | 2025-05-09 09:15:00 | 1335.03 | PARTIAL | 0.50 | 5.11% |
| SELL | retest2 | 2025-04-28 09:45:00 | 1408.60 | 2025-05-09 09:15:00 | 1336.56 | PARTIAL | 0.50 | 5.11% |
| SELL | retest2 | 2025-04-21 11:15:00 | 1424.00 | 2025-05-13 09:15:00 | 1396.10 | STOP_HIT | 0.50 | 1.96% |
| SELL | retest2 | 2025-04-25 10:15:00 | 1407.00 | 2025-05-13 09:15:00 | 1396.10 | STOP_HIT | 0.50 | 0.77% |
| SELL | retest2 | 2025-04-25 10:45:00 | 1405.30 | 2025-05-13 09:15:00 | 1396.10 | STOP_HIT | 0.50 | 0.65% |
| SELL | retest2 | 2025-04-28 09:15:00 | 1406.90 | 2025-05-13 09:15:00 | 1396.10 | STOP_HIT | 0.50 | 0.77% |
| SELL | retest2 | 2025-04-28 09:45:00 | 1408.60 | 2025-05-13 09:15:00 | 1396.10 | STOP_HIT | 0.50 | 0.89% |
| SELL | retest2 | 2025-05-02 09:15:00 | 1377.70 | 2025-05-19 09:15:00 | 1452.60 | STOP_HIT | 1.00 | -5.44% |
| SELL | retest2 | 2025-05-02 10:00:00 | 1378.70 | 2025-05-19 09:15:00 | 1452.60 | STOP_HIT | 1.00 | -5.36% |
| SELL | retest2 | 2025-05-02 10:45:00 | 1376.00 | 2025-05-19 09:15:00 | 1452.60 | STOP_HIT | 1.00 | -5.57% |
| SELL | retest2 | 2025-05-02 14:15:00 | 1378.30 | 2025-05-19 09:15:00 | 1452.60 | STOP_HIT | 1.00 | -5.39% |
| SELL | retest2 | 2025-05-15 11:30:00 | 1400.40 | 2025-05-19 09:15:00 | 1452.60 | STOP_HIT | 1.00 | -3.73% |
| SELL | retest2 | 2025-05-15 12:30:00 | 1396.70 | 2025-05-19 09:15:00 | 1452.60 | STOP_HIT | 1.00 | -4.00% |
| SELL | retest2 | 2025-05-15 13:15:00 | 1399.30 | 2025-05-19 09:15:00 | 1452.60 | STOP_HIT | 1.00 | -3.81% |
| SELL | retest2 | 2025-05-16 10:00:00 | 1397.20 | 2025-05-19 09:15:00 | 1452.60 | STOP_HIT | 1.00 | -3.97% |
| SELL | retest2 | 2025-06-12 11:30:00 | 1401.90 | 2025-06-19 12:15:00 | 1331.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-12 12:45:00 | 1403.30 | 2025-06-19 12:15:00 | 1333.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-12 11:30:00 | 1401.90 | 2025-06-30 13:15:00 | 1379.30 | STOP_HIT | 0.50 | 1.61% |
| SELL | retest2 | 2025-06-12 12:45:00 | 1403.30 | 2025-06-30 13:15:00 | 1379.30 | STOP_HIT | 0.50 | 1.71% |
| BUY | retest2 | 2025-08-04 12:15:00 | 1444.50 | 2025-08-05 12:15:00 | 1416.00 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-08-04 13:00:00 | 1440.20 | 2025-08-05 12:15:00 | 1416.00 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-08-04 14:45:00 | 1442.10 | 2025-08-05 12:15:00 | 1416.00 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-08-25 11:30:00 | 1392.30 | 2025-08-26 14:15:00 | 1435.00 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2025-08-26 09:15:00 | 1380.70 | 2025-08-26 14:15:00 | 1435.00 | STOP_HIT | 1.00 | -3.93% |
| SELL | retest2 | 2025-08-29 10:15:00 | 1388.50 | 2025-09-08 13:15:00 | 1319.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-29 10:15:00 | 1388.50 | 2025-09-19 13:15:00 | 1361.30 | STOP_HIT | 0.50 | 1.96% |
| SELL | retest2 | 2025-09-22 09:15:00 | 1390.30 | 2025-09-26 09:15:00 | 1320.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 09:15:00 | 1390.30 | 2025-09-26 12:15:00 | 1374.50 | STOP_HIT | 0.50 | 1.14% |
| SELL | retest2 | 2025-09-29 09:15:00 | 1351.00 | 2025-10-07 12:15:00 | 1388.40 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2025-09-29 10:15:00 | 1347.20 | 2025-10-07 12:15:00 | 1388.40 | STOP_HIT | 1.00 | -3.06% |
| SELL | retest2 | 2025-09-30 12:15:00 | 1350.00 | 2025-10-07 12:15:00 | 1388.40 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2025-10-08 09:15:00 | 1351.20 | 2025-10-16 12:15:00 | 1283.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-08 09:15:00 | 1351.20 | 2025-11-03 12:15:00 | 1312.40 | STOP_HIT | 0.50 | 2.87% |
| SELL | retest2 | 2025-11-13 13:15:00 | 1323.10 | 2025-11-14 09:15:00 | 1349.10 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-11-13 13:45:00 | 1327.50 | 2025-11-14 09:15:00 | 1349.10 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-12-18 15:00:00 | 1399.70 | 2026-01-08 12:15:00 | 1539.67 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-19 13:30:00 | 1406.10 | 2026-01-08 12:15:00 | 1546.71 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-26 13:15:00 | 1400.00 | 2026-01-08 12:15:00 | 1540.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-31 10:15:00 | 1413.40 | 2026-01-09 13:15:00 | 1554.74 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-31 11:30:00 | 1429.50 | 2026-01-09 13:15:00 | 1572.45 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-01 10:00:00 | 1425.70 | 2026-01-09 13:15:00 | 1568.27 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-01 10:45:00 | 1425.70 | 2026-01-09 13:15:00 | 1568.27 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-06 09:15:00 | 1432.50 | 2026-01-09 13:15:00 | 1575.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-27 10:15:00 | 1474.50 | 2026-02-01 11:15:00 | 1424.50 | STOP_HIT | 1.00 | -3.39% |
| BUY | retest2 | 2026-01-27 11:15:00 | 1468.60 | 2026-02-01 11:15:00 | 1424.50 | STOP_HIT | 1.00 | -3.00% |
| BUY | retest2 | 2026-01-28 10:30:00 | 1472.60 | 2026-02-01 11:15:00 | 1424.50 | STOP_HIT | 1.00 | -3.27% |
| BUY | retest2 | 2026-01-29 11:45:00 | 1472.70 | 2026-02-01 11:15:00 | 1424.50 | STOP_HIT | 1.00 | -3.27% |
| BUY | retest2 | 2026-02-11 13:00:00 | 1469.80 | 2026-02-11 13:15:00 | 1436.00 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2026-02-12 14:45:00 | 1468.00 | 2026-02-13 09:15:00 | 1433.70 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2026-02-13 12:30:00 | 1474.00 | 2026-03-17 09:15:00 | 1621.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-13 13:00:00 | 1475.70 | 2026-03-17 09:15:00 | 1623.27 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-23 09:15:00 | 1479.10 | 2026-03-17 09:15:00 | 1608.20 | TARGET_HIT | 1.00 | 8.73% |
| BUY | retest2 | 2026-03-09 10:45:00 | 1462.00 | 2026-04-07 14:15:00 | 1441.60 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2026-04-02 11:45:00 | 1461.50 | 2026-04-07 14:15:00 | 1441.60 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2026-04-06 09:30:00 | 1457.60 | 2026-04-07 14:15:00 | 1441.60 | STOP_HIT | 1.00 | -1.10% |
