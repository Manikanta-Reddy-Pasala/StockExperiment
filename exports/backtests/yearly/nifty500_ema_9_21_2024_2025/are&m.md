# Amara Raja Energy & Mobility Ltd. (ARE&M)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 890.85
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 127 |
| ALERT1 | 83 |
| ALERT2 | 84 |
| ALERT2_SKIP | 42 |
| ALERT3 | 220 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 125 |
| PARTIAL | 19 |
| TARGET_HIT | 10 |
| STOP_HIT | 118 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 147 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 63 / 84
- **Target hits / Stop hits / Partials:** 10 / 118 / 19
- **Avg / median % per leg:** 0.81% / -0.71%
- **Sum % (uncompounded):** 119.23%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 48 | 12 | 25.0% | 3 | 45 | 0 | -0.36% | -17.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 48 | 12 | 25.0% | 3 | 45 | 0 | -0.36% | -17.3% |
| SELL (all) | 99 | 51 | 51.5% | 7 | 73 | 19 | 1.38% | 136.5% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.50% | -4.5% |
| SELL @ 3rd Alert (retest2) | 96 | 51 | 53.1% | 7 | 70 | 19 | 1.47% | 141.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.50% | -4.5% |
| retest2 (combined) | 144 | 63 | 43.8% | 10 | 115 | 19 | 0.86% | 123.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 09:15:00 | 1105.25 | 1084.08 | 1081.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 15:15:00 | 1123.00 | 1107.09 | 1095.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 14:15:00 | 1155.55 | 1159.45 | 1149.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-21 15:00:00 | 1155.55 | 1159.45 | 1149.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 15:15:00 | 1153.00 | 1158.16 | 1149.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:30:00 | 1131.30 | 1155.53 | 1149.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 1147.85 | 1153.99 | 1149.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 10:30:00 | 1144.65 | 1153.99 | 1149.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 11:15:00 | 1152.70 | 1153.73 | 1149.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 11:30:00 | 1144.40 | 1153.73 | 1149.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 12:15:00 | 1144.20 | 1151.83 | 1149.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 12:30:00 | 1142.00 | 1151.83 | 1149.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 13:15:00 | 1137.55 | 1148.97 | 1148.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 13:30:00 | 1140.10 | 1148.97 | 1148.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2024-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 15:15:00 | 1142.60 | 1146.66 | 1147.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-23 09:15:00 | 1121.00 | 1141.53 | 1144.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-24 09:15:00 | 1163.40 | 1132.08 | 1135.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 09:15:00 | 1163.40 | 1132.08 | 1135.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 1163.40 | 1132.08 | 1135.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 10:00:00 | 1163.40 | 1132.08 | 1135.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2024-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 10:15:00 | 1186.00 | 1142.86 | 1140.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 11:15:00 | 1208.00 | 1155.89 | 1146.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-27 14:15:00 | 1217.75 | 1219.51 | 1195.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-27 15:00:00 | 1217.75 | 1219.51 | 1195.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 1244.70 | 1241.61 | 1223.61 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2024-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 11:15:00 | 1205.00 | 1217.63 | 1218.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 13:15:00 | 1199.75 | 1211.27 | 1215.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 13:15:00 | 1200.00 | 1198.73 | 1205.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 13:15:00 | 1200.00 | 1198.73 | 1205.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 1200.00 | 1198.73 | 1205.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 14:00:00 | 1200.00 | 1198.73 | 1205.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 1201.95 | 1193.95 | 1201.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 10:45:00 | 1193.05 | 1193.68 | 1200.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 12:45:00 | 1194.60 | 1194.66 | 1199.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 15:15:00 | 1194.00 | 1195.81 | 1199.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 1133.40 | 1185.54 | 1194.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 1134.87 | 1185.54 | 1194.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 1134.30 | 1185.54 | 1194.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-06-04 10:15:00 | 1073.74 | 1165.95 | 1184.35 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 12:15:00 | 1211.90 | 1161.81 | 1161.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 1296.25 | 1209.80 | 1185.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 14:15:00 | 1402.20 | 1406.80 | 1358.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 14:30:00 | 1398.90 | 1406.80 | 1358.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 1381.85 | 1401.04 | 1364.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:30:00 | 1379.00 | 1401.04 | 1364.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 12:15:00 | 1361.30 | 1384.35 | 1364.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 12:45:00 | 1353.35 | 1384.35 | 1364.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 13:15:00 | 1368.00 | 1381.08 | 1365.26 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2024-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 14:15:00 | 1344.55 | 1359.91 | 1361.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-14 11:15:00 | 1333.95 | 1343.79 | 1350.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-18 09:15:00 | 1365.00 | 1345.18 | 1347.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-18 09:15:00 | 1365.00 | 1345.18 | 1347.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 1365.00 | 1345.18 | 1347.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 09:30:00 | 1365.20 | 1345.18 | 1347.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2024-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 10:15:00 | 1389.95 | 1354.13 | 1351.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 09:15:00 | 1421.10 | 1407.01 | 1396.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 12:15:00 | 1406.40 | 1408.24 | 1399.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-21 13:00:00 | 1406.40 | 1408.24 | 1399.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 1394.00 | 1405.92 | 1401.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:30:00 | 1385.95 | 1405.92 | 1401.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 1397.00 | 1404.13 | 1400.96 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2024-06-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 14:15:00 | 1380.45 | 1396.56 | 1398.18 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 09:15:00 | 1595.75 | 1433.75 | 1414.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 09:15:00 | 1674.80 | 1597.18 | 1521.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 09:15:00 | 1618.70 | 1651.64 | 1594.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 13:15:00 | 1608.00 | 1627.84 | 1600.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 1608.00 | 1627.84 | 1600.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 13:45:00 | 1606.95 | 1627.84 | 1600.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 1608.00 | 1621.71 | 1602.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:15:00 | 1599.00 | 1621.71 | 1602.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 1604.45 | 1618.25 | 1602.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 11:45:00 | 1690.00 | 1631.91 | 1611.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-09 09:15:00 | 1676.95 | 1687.31 | 1688.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 09:15:00 | 1676.95 | 1687.31 | 1688.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-09 10:15:00 | 1664.00 | 1682.65 | 1685.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 13:15:00 | 1660.05 | 1652.49 | 1663.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-10 14:00:00 | 1660.05 | 1652.49 | 1663.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 14:15:00 | 1685.10 | 1659.02 | 1665.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 15:00:00 | 1685.10 | 1659.02 | 1665.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 15:15:00 | 1684.80 | 1664.17 | 1667.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:15:00 | 1667.10 | 1664.17 | 1667.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 1623.80 | 1650.11 | 1657.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 15:00:00 | 1616.80 | 1630.09 | 1643.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 09:15:00 | 1593.20 | 1627.88 | 1641.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 14:15:00 | 1535.96 | 1554.69 | 1570.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 09:15:00 | 1513.54 | 1548.24 | 1564.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-23 09:15:00 | 1538.70 | 1537.52 | 1549.90 | SL hit (close>ema200) qty=0.50 sl=1537.52 alert=retest2 |

### Cycle 11 — BUY (started 2024-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 10:15:00 | 1601.95 | 1557.81 | 1554.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 10:15:00 | 1607.70 | 1581.35 | 1572.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 11:15:00 | 1642.50 | 1643.76 | 1617.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 12:00:00 | 1642.50 | 1643.76 | 1617.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 1630.40 | 1640.29 | 1625.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 09:45:00 | 1627.80 | 1640.29 | 1625.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 10:15:00 | 1678.60 | 1647.95 | 1630.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 11:00:00 | 1678.60 | 1647.95 | 1630.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 15:15:00 | 1648.75 | 1653.34 | 1640.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 09:15:00 | 1645.00 | 1653.34 | 1640.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 1632.65 | 1649.20 | 1639.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 10:00:00 | 1632.65 | 1649.20 | 1639.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 1626.25 | 1644.61 | 1638.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 11:00:00 | 1626.25 | 1644.61 | 1638.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 1628.35 | 1635.89 | 1635.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 15:00:00 | 1628.35 | 1635.89 | 1635.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2024-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 15:15:00 | 1626.00 | 1633.91 | 1634.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 11:15:00 | 1617.55 | 1630.65 | 1633.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 10:15:00 | 1612.05 | 1611.63 | 1620.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-02 11:00:00 | 1612.05 | 1611.63 | 1620.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 13:15:00 | 1617.00 | 1612.47 | 1618.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 14:00:00 | 1617.00 | 1612.47 | 1618.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 14:15:00 | 1613.65 | 1612.70 | 1618.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-05 09:15:00 | 1539.15 | 1612.76 | 1617.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-06 15:15:00 | 1462.19 | 1494.72 | 1529.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-07 12:15:00 | 1495.70 | 1490.26 | 1515.55 | SL hit (close>ema200) qty=0.50 sl=1490.26 alert=retest2 |

### Cycle 13 — BUY (started 2024-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 09:15:00 | 1576.70 | 1521.06 | 1517.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 10:15:00 | 1607.90 | 1538.43 | 1525.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 12:15:00 | 1580.90 | 1597.46 | 1573.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-13 13:00:00 | 1580.90 | 1597.46 | 1573.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 1570.10 | 1591.98 | 1573.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:00:00 | 1570.10 | 1591.98 | 1573.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 1565.50 | 1586.69 | 1572.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 15:00:00 | 1565.50 | 1586.69 | 1572.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 1569.95 | 1583.34 | 1572.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:15:00 | 1538.05 | 1583.34 | 1572.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 1542.00 | 1575.07 | 1569.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:30:00 | 1532.40 | 1575.07 | 1569.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 1534.85 | 1567.03 | 1566.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 11:00:00 | 1534.85 | 1567.03 | 1566.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2024-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 11:15:00 | 1541.00 | 1561.82 | 1564.08 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 1565.20 | 1560.31 | 1560.10 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 10:15:00 | 1557.85 | 1559.82 | 1559.90 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 11:15:00 | 1563.75 | 1560.61 | 1560.25 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 13:15:00 | 1556.85 | 1559.56 | 1559.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-20 09:15:00 | 1540.60 | 1555.64 | 1557.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-20 15:15:00 | 1545.00 | 1544.79 | 1550.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-21 09:15:00 | 1560.10 | 1544.79 | 1550.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 1557.05 | 1547.24 | 1550.75 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2024-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 09:15:00 | 1568.05 | 1554.78 | 1553.20 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-08-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 09:15:00 | 1535.50 | 1550.92 | 1552.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 14:15:00 | 1526.50 | 1537.68 | 1544.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 09:15:00 | 1542.85 | 1528.11 | 1534.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 09:15:00 | 1542.85 | 1528.11 | 1534.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 1542.85 | 1528.11 | 1534.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 09:30:00 | 1552.00 | 1528.11 | 1534.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 1540.20 | 1530.53 | 1534.66 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2024-08-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 11:15:00 | 1590.40 | 1542.50 | 1539.73 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 15:15:00 | 1537.00 | 1546.95 | 1547.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 09:15:00 | 1527.85 | 1543.13 | 1545.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 12:15:00 | 1508.80 | 1506.81 | 1512.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 12:15:00 | 1508.80 | 1506.81 | 1512.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 12:15:00 | 1508.80 | 1506.81 | 1512.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 12:45:00 | 1510.45 | 1506.81 | 1512.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 1508.05 | 1507.58 | 1511.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 09:15:00 | 1506.80 | 1507.58 | 1511.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 10:30:00 | 1505.60 | 1506.76 | 1510.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 09:15:00 | 1431.46 | 1485.83 | 1498.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 09:15:00 | 1430.32 | 1485.83 | 1498.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-09 14:15:00 | 1412.00 | 1408.59 | 1428.26 | SL hit (close>ema200) qty=0.50 sl=1408.59 alert=retest2 |

### Cycle 23 — BUY (started 2024-09-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 13:15:00 | 1413.50 | 1412.96 | 1412.93 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-09-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 14:15:00 | 1411.90 | 1412.75 | 1412.84 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 09:15:00 | 1420.65 | 1414.37 | 1413.56 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 11:15:00 | 1400.65 | 1411.24 | 1412.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 14:15:00 | 1399.60 | 1405.30 | 1407.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 15:15:00 | 1377.00 | 1374.90 | 1385.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 09:15:00 | 1367.65 | 1374.90 | 1385.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 1378.70 | 1375.66 | 1384.57 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 10:15:00 | 1399.10 | 1386.88 | 1386.66 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-23 13:15:00 | 1374.85 | 1385.30 | 1386.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-24 12:15:00 | 1372.75 | 1379.13 | 1382.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 14:15:00 | 1368.20 | 1365.99 | 1371.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-25 15:00:00 | 1368.20 | 1365.99 | 1371.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 1349.80 | 1362.59 | 1369.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 11:15:00 | 1345.15 | 1359.98 | 1367.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 10:45:00 | 1344.05 | 1340.15 | 1351.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-27 12:15:00 | 1391.85 | 1353.68 | 1355.88 | SL hit (close>static) qty=1.00 sl=1369.70 alert=retest2 |

### Cycle 29 — BUY (started 2024-09-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 13:15:00 | 1380.55 | 1359.06 | 1358.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 15:15:00 | 1397.95 | 1371.79 | 1364.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 09:15:00 | 1366.15 | 1370.66 | 1364.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 09:15:00 | 1366.15 | 1370.66 | 1364.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 1366.15 | 1370.66 | 1364.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:45:00 | 1362.90 | 1370.66 | 1364.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 1382.90 | 1373.11 | 1366.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 12:30:00 | 1392.10 | 1380.16 | 1370.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 11:15:00 | 1359.05 | 1391.43 | 1391.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 11:15:00 | 1359.05 | 1391.43 | 1391.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 13:15:00 | 1350.85 | 1377.62 | 1385.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 09:15:00 | 1375.10 | 1371.46 | 1379.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 09:15:00 | 1375.10 | 1371.46 | 1379.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 1375.10 | 1371.46 | 1379.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:45:00 | 1378.40 | 1371.46 | 1379.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 1384.95 | 1374.16 | 1380.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 10:45:00 | 1386.90 | 1374.16 | 1380.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 1388.60 | 1377.05 | 1381.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 11:30:00 | 1388.95 | 1377.05 | 1381.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 13:15:00 | 1383.00 | 1378.05 | 1380.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 13:45:00 | 1386.25 | 1378.05 | 1380.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 1383.55 | 1379.15 | 1381.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 15:15:00 | 1373.00 | 1379.15 | 1381.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 15:15:00 | 1373.00 | 1377.92 | 1380.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-07 09:15:00 | 1362.75 | 1377.92 | 1380.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 1337.90 | 1369.92 | 1376.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 10:15:00 | 1331.10 | 1369.92 | 1376.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 09:30:00 | 1327.00 | 1332.98 | 1350.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 10:00:00 | 1324.10 | 1332.98 | 1350.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-09 11:15:00 | 1382.30 | 1351.81 | 1348.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 11:15:00 | 1382.30 | 1351.81 | 1348.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 14:15:00 | 1401.15 | 1384.93 | 1372.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-14 09:15:00 | 1393.00 | 1400.79 | 1390.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-14 10:00:00 | 1393.00 | 1400.79 | 1390.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 1390.35 | 1398.70 | 1390.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 11:00:00 | 1390.35 | 1398.70 | 1390.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 11:15:00 | 1392.00 | 1397.36 | 1390.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 12:15:00 | 1390.00 | 1397.36 | 1390.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 12:15:00 | 1392.90 | 1396.47 | 1390.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 12:45:00 | 1391.55 | 1396.47 | 1390.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 13:15:00 | 1390.00 | 1395.17 | 1390.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 13:30:00 | 1391.15 | 1395.17 | 1390.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 14:15:00 | 1391.35 | 1394.41 | 1390.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 09:15:00 | 1399.40 | 1393.94 | 1390.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-15 09:15:00 | 1386.75 | 1392.50 | 1390.54 | SL hit (close<static) qty=1.00 sl=1388.50 alert=retest2 |

### Cycle 32 — SELL (started 2024-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 09:15:00 | 1372.90 | 1390.05 | 1391.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 1347.75 | 1369.93 | 1378.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 10:15:00 | 1289.75 | 1283.27 | 1305.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 11:00:00 | 1289.75 | 1283.27 | 1305.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 1289.60 | 1253.71 | 1263.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 10:00:00 | 1289.60 | 1253.71 | 1263.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 10:15:00 | 1307.20 | 1264.40 | 1267.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 11:00:00 | 1307.20 | 1264.40 | 1267.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2024-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 11:15:00 | 1334.60 | 1278.44 | 1273.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 10:15:00 | 1366.05 | 1350.92 | 1334.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 1356.85 | 1378.99 | 1361.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 1356.85 | 1378.99 | 1361.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1356.85 | 1378.99 | 1361.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 1356.85 | 1378.99 | 1361.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 1353.20 | 1373.83 | 1360.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 1353.20 | 1373.83 | 1360.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 1366.30 | 1372.33 | 1360.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 14:00:00 | 1373.40 | 1371.40 | 1362.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-05 09:15:00 | 1326.80 | 1363.29 | 1361.10 | SL hit (close<static) qty=1.00 sl=1348.00 alert=retest2 |

### Cycle 34 — SELL (started 2024-11-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 10:15:00 | 1323.00 | 1355.23 | 1357.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 10:15:00 | 1307.50 | 1316.58 | 1324.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 12:15:00 | 1315.60 | 1314.19 | 1322.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-08 13:00:00 | 1315.60 | 1314.19 | 1322.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 1282.00 | 1292.74 | 1303.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 11:15:00 | 1280.35 | 1290.54 | 1301.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-19 12:15:00 | 1271.05 | 1252.95 | 1252.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2024-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 12:15:00 | 1271.05 | 1252.95 | 1252.90 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 15:15:00 | 1248.00 | 1253.11 | 1253.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 09:15:00 | 1235.95 | 1249.68 | 1251.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 1235.05 | 1235.00 | 1241.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-22 10:00:00 | 1235.05 | 1235.00 | 1241.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 1245.00 | 1235.11 | 1238.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 09:30:00 | 1267.50 | 1235.11 | 1238.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2024-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 11:15:00 | 1250.05 | 1241.31 | 1240.70 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 10:15:00 | 1232.20 | 1239.64 | 1240.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-26 11:15:00 | 1227.05 | 1237.12 | 1239.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 09:15:00 | 1239.40 | 1234.89 | 1236.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 09:15:00 | 1239.40 | 1234.89 | 1236.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 1239.40 | 1234.89 | 1236.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:45:00 | 1252.80 | 1234.89 | 1236.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2024-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 10:15:00 | 1252.60 | 1238.43 | 1238.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 13:15:00 | 1263.00 | 1247.88 | 1243.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 11:15:00 | 1265.00 | 1265.28 | 1259.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-29 11:45:00 | 1263.50 | 1265.28 | 1259.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 1272.75 | 1272.43 | 1265.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 15:15:00 | 1295.00 | 1281.33 | 1273.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-10 10:15:00 | 1312.60 | 1324.28 | 1325.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2024-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 10:15:00 | 1312.60 | 1324.28 | 1325.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 14:15:00 | 1309.30 | 1318.64 | 1322.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 12:15:00 | 1318.75 | 1313.81 | 1317.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 12:15:00 | 1318.75 | 1313.81 | 1317.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 12:15:00 | 1318.75 | 1313.81 | 1317.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 12:45:00 | 1319.05 | 1313.81 | 1317.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 13:15:00 | 1318.95 | 1314.84 | 1318.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 13:45:00 | 1318.00 | 1314.84 | 1318.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 14:15:00 | 1316.00 | 1315.07 | 1317.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 15:00:00 | 1316.00 | 1315.07 | 1317.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 15:15:00 | 1313.80 | 1314.82 | 1317.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 09:15:00 | 1308.55 | 1314.82 | 1317.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 12:15:00 | 1243.12 | 1258.43 | 1270.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-12-23 12:15:00 | 1177.69 | 1197.19 | 1215.53 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 41 — BUY (started 2024-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 10:15:00 | 1234.70 | 1204.44 | 1202.66 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 14:15:00 | 1181.40 | 1206.99 | 1209.05 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 15:15:00 | 1213.50 | 1204.39 | 1203.48 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 11:15:00 | 1193.50 | 1202.04 | 1202.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 14:15:00 | 1189.45 | 1194.69 | 1198.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 09:15:00 | 1125.95 | 1120.38 | 1137.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-09 09:45:00 | 1127.90 | 1120.38 | 1137.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 1076.35 | 1080.85 | 1097.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-13 09:30:00 | 1086.90 | 1080.85 | 1097.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 1053.45 | 1055.47 | 1070.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 12:15:00 | 1052.00 | 1055.47 | 1070.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 10:00:00 | 1051.50 | 1055.63 | 1064.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 13:15:00 | 1051.00 | 1057.14 | 1063.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 15:15:00 | 1052.00 | 1054.91 | 1061.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 1074.60 | 1058.39 | 1061.67 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-16 09:15:00 | 1074.60 | 1058.39 | 1061.67 | SL hit (close>static) qty=1.00 sl=1071.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 11:15:00 | 1073.30 | 1063.92 | 1063.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 14:15:00 | 1077.45 | 1069.00 | 1066.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 09:15:00 | 1068.80 | 1069.60 | 1067.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 09:15:00 | 1068.80 | 1069.60 | 1067.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 1068.80 | 1069.60 | 1067.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 14:15:00 | 1077.15 | 1068.75 | 1067.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-22 10:15:00 | 1065.15 | 1085.95 | 1087.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 10:15:00 | 1065.15 | 1085.95 | 1087.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 1060.95 | 1080.95 | 1085.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 1081.70 | 1074.17 | 1079.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 1081.70 | 1074.17 | 1079.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 1081.70 | 1074.17 | 1079.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 1081.70 | 1074.17 | 1079.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 1091.75 | 1077.68 | 1080.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:45:00 | 1089.30 | 1077.68 | 1080.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 12:15:00 | 1093.75 | 1082.85 | 1082.38 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-23 15:15:00 | 1075.60 | 1081.21 | 1081.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 10:15:00 | 1069.30 | 1079.45 | 1080.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 11:15:00 | 1033.35 | 1021.96 | 1038.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 12:00:00 | 1033.35 | 1021.96 | 1038.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 1041.90 | 1025.95 | 1039.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 12:45:00 | 1035.70 | 1025.95 | 1039.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 1034.70 | 1027.70 | 1038.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 14:45:00 | 1029.05 | 1028.41 | 1038.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 09:15:00 | 1063.50 | 1036.16 | 1039.91 | SL hit (close>static) qty=1.00 sl=1043.55 alert=retest2 |

### Cycle 49 — BUY (started 2025-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 09:15:00 | 1087.00 | 1044.05 | 1039.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 13:15:00 | 1090.85 | 1078.46 | 1071.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 10:15:00 | 1081.40 | 1083.76 | 1076.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 11:00:00 | 1081.40 | 1083.76 | 1076.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 1079.00 | 1082.93 | 1079.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:15:00 | 1072.25 | 1082.93 | 1079.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 1075.00 | 1081.35 | 1078.81 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2025-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 13:15:00 | 1067.70 | 1076.08 | 1076.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 1035.55 | 1066.29 | 1072.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 12:15:00 | 1005.30 | 1002.82 | 1018.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 12:45:00 | 1004.85 | 1002.82 | 1018.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 965.80 | 962.33 | 970.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 11:00:00 | 957.00 | 961.26 | 968.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 10:00:00 | 954.25 | 952.76 | 960.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 14:15:00 | 974.30 | 958.29 | 960.06 | SL hit (close>static) qty=1.00 sl=971.55 alert=retest2 |

### Cycle 51 — BUY (started 2025-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 09:15:00 | 998.50 | 968.04 | 964.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 10:15:00 | 1025.70 | 979.57 | 969.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 10:15:00 | 1020.00 | 1024.67 | 1004.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 11:00:00 | 1020.00 | 1024.67 | 1004.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 1008.00 | 1017.44 | 1009.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 13:00:00 | 1017.20 | 1014.34 | 1009.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 09:15:00 | 1025.15 | 1011.03 | 1009.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-27 09:45:00 | 1018.00 | 1033.13 | 1025.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-27 12:30:00 | 1017.80 | 1024.49 | 1023.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-27 13:15:00 | 1002.85 | 1020.16 | 1021.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-02-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 13:15:00 | 1002.85 | 1020.16 | 1021.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 14:15:00 | 998.85 | 1015.90 | 1019.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 09:15:00 | 964.15 | 955.80 | 972.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-04 09:15:00 | 964.15 | 955.80 | 972.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 964.15 | 955.80 | 972.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 964.15 | 955.80 | 972.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 969.75 | 958.59 | 971.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:00:00 | 969.75 | 958.59 | 971.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 11:15:00 | 968.55 | 960.58 | 971.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 13:15:00 | 961.85 | 962.47 | 971.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 14:30:00 | 962.90 | 965.21 | 971.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 15:15:00 | 962.20 | 965.21 | 971.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 997.85 | 971.25 | 972.90 | SL hit (close>static) qty=1.00 sl=972.60 alert=retest2 |

### Cycle 53 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 988.20 | 974.64 | 974.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 14:15:00 | 1003.95 | 987.58 | 981.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 15:15:00 | 1000.00 | 1002.18 | 994.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 09:15:00 | 1004.80 | 1002.18 | 994.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 1008.20 | 1003.39 | 995.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 11:15:00 | 1013.30 | 1005.08 | 996.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 10:00:00 | 1019.40 | 1005.58 | 1000.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 11:15:00 | 1015.80 | 1006.67 | 1001.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 12:45:00 | 1017.55 | 1009.79 | 1003.66 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 1003.95 | 1008.93 | 1004.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:30:00 | 1003.70 | 1008.93 | 1004.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 994.00 | 1005.94 | 1003.41 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-11 09:15:00 | 984.90 | 1001.73 | 1001.73 | SL hit (close<static) qty=1.00 sl=992.00 alert=retest2 |

### Cycle 54 — SELL (started 2025-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 10:15:00 | 990.25 | 999.44 | 1000.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 11:15:00 | 976.50 | 986.10 | 991.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 09:15:00 | 987.80 | 982.58 | 987.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 09:15:00 | 987.80 | 982.58 | 987.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 987.80 | 982.58 | 987.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:00:00 | 987.80 | 982.58 | 987.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 978.80 | 981.82 | 986.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 12:45:00 | 975.00 | 978.80 | 984.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 11:00:00 | 974.95 | 972.26 | 978.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 14:15:00 | 973.75 | 973.57 | 977.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 991.60 | 976.21 | 977.57 | SL hit (close>static) qty=1.00 sl=990.45 alert=retest2 |

### Cycle 55 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 993.65 | 979.70 | 979.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 11:15:00 | 999.00 | 983.56 | 980.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 1034.15 | 1036.20 | 1017.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 10:00:00 | 1034.15 | 1036.20 | 1017.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 1045.60 | 1040.16 | 1028.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 10:15:00 | 1053.65 | 1040.16 | 1028.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 14:15:00 | 1040.00 | 1058.28 | 1058.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2025-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 14:15:00 | 1040.00 | 1058.28 | 1058.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 15:15:00 | 1034.80 | 1053.59 | 1056.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 13:15:00 | 1015.40 | 1014.08 | 1023.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 13:15:00 | 1015.40 | 1014.08 | 1023.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 13:15:00 | 1015.40 | 1014.08 | 1023.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 13:45:00 | 1019.30 | 1014.08 | 1023.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 1020.35 | 1015.61 | 1020.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:00:00 | 1020.35 | 1015.61 | 1020.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 1022.45 | 1016.98 | 1020.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:00:00 | 1022.45 | 1016.98 | 1020.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 1027.60 | 1019.10 | 1021.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:30:00 | 1024.00 | 1019.10 | 1021.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 1030.45 | 1021.37 | 1022.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 15:00:00 | 1030.45 | 1021.37 | 1022.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 986.15 | 1009.91 | 1015.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 10:15:00 | 980.75 | 1009.91 | 1015.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 13:00:00 | 983.55 | 998.12 | 1008.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 09:15:00 | 936.25 | 994.44 | 1003.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-07 09:15:00 | 882.68 | 981.13 | 996.86 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 981.50 | 972.37 | 971.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 998.10 | 979.58 | 975.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 14:15:00 | 994.90 | 995.65 | 990.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 15:00:00 | 994.90 | 995.65 | 990.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 13:15:00 | 1019.60 | 1026.08 | 1021.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 13:30:00 | 1020.20 | 1026.08 | 1021.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 14:15:00 | 1017.70 | 1024.41 | 1021.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 15:00:00 | 1017.70 | 1024.41 | 1021.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2025-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 10:15:00 | 1008.20 | 1017.48 | 1018.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-24 09:15:00 | 1000.90 | 1009.59 | 1013.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-25 15:15:00 | 988.50 | 988.40 | 996.10 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-28 09:15:00 | 978.90 | 988.40 | 996.10 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-28 10:15:00 | 983.70 | 988.92 | 995.63 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-28 14:15:00 | 982.00 | 987.20 | 992.58 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 996.30 | 988.48 | 991.76 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-29 09:15:00 | 996.30 | 988.48 | 991.76 | SL hit (close>ema400) qty=1.00 sl=991.76 alert=retest1 |

### Cycle 59 — BUY (started 2025-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 12:15:00 | 987.10 | 964.19 | 961.53 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 944.45 | 959.10 | 960.11 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 987.75 | 961.46 | 959.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 10:15:00 | 996.00 | 968.37 | 963.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 12:15:00 | 1009.50 | 1011.98 | 1000.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 12:45:00 | 1008.60 | 1011.98 | 1000.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 1035.00 | 1037.22 | 1031.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:45:00 | 1034.20 | 1037.22 | 1031.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1023.15 | 1033.91 | 1030.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 1024.40 | 1033.91 | 1030.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 1025.00 | 1032.13 | 1030.17 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 1021.75 | 1028.73 | 1028.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 1013.00 | 1025.58 | 1027.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 14:15:00 | 1015.20 | 1014.40 | 1019.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 15:00:00 | 1015.20 | 1014.40 | 1019.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 1016.95 | 1014.68 | 1018.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 13:45:00 | 1008.85 | 1013.43 | 1016.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 11:15:00 | 1027.05 | 1017.92 | 1017.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2025-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 11:15:00 | 1027.05 | 1017.92 | 1017.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 1048.95 | 1026.97 | 1022.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 1039.25 | 1042.18 | 1034.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 10:00:00 | 1039.25 | 1042.18 | 1034.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 1039.00 | 1043.83 | 1037.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 13:30:00 | 1037.30 | 1043.83 | 1037.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 1047.00 | 1044.47 | 1038.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 10:45:00 | 1049.65 | 1045.88 | 1040.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 11:30:00 | 1056.50 | 1048.41 | 1042.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 12:15:00 | 1037.90 | 1052.43 | 1053.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 12:15:00 | 1037.90 | 1052.43 | 1053.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 14:15:00 | 1032.80 | 1045.84 | 1050.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 11:15:00 | 1009.20 | 1008.48 | 1017.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 11:30:00 | 1010.80 | 1008.48 | 1017.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 1001.50 | 1001.67 | 1006.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:45:00 | 1005.40 | 1001.67 | 1006.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 1005.60 | 1001.85 | 1004.02 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2025-06-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 12:15:00 | 1013.00 | 1005.23 | 1005.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-12 10:15:00 | 1022.70 | 1016.03 | 1013.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 12:15:00 | 1014.20 | 1015.68 | 1013.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 12:15:00 | 1014.20 | 1015.68 | 1013.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 1014.20 | 1015.68 | 1013.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 13:15:00 | 1011.90 | 1015.68 | 1013.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 1003.00 | 1013.14 | 1012.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 1003.00 | 1013.14 | 1012.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 1007.50 | 1012.01 | 1012.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 10:15:00 | 998.30 | 1006.90 | 1009.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 15:15:00 | 1006.80 | 1004.17 | 1006.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 15:15:00 | 1006.80 | 1004.17 | 1006.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 1006.80 | 1004.17 | 1006.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:15:00 | 993.00 | 1004.17 | 1006.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 11:15:00 | 1000.50 | 1002.29 | 1005.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 11:45:00 | 1001.00 | 1002.15 | 1005.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 12:15:00 | 1001.20 | 1002.15 | 1005.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 1004.60 | 1002.31 | 1004.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 1004.60 | 1002.31 | 1004.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 1004.40 | 1002.73 | 1004.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 998.00 | 1002.73 | 1004.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 10:15:00 | 977.20 | 973.51 | 973.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2025-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 10:15:00 | 977.20 | 973.51 | 973.37 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 13:15:00 | 970.20 | 973.31 | 973.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 14:15:00 | 961.20 | 970.89 | 972.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 11:15:00 | 974.90 | 970.14 | 971.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 11:15:00 | 974.90 | 970.14 | 971.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 11:15:00 | 974.90 | 970.14 | 971.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 12:00:00 | 974.90 | 970.14 | 971.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 12:15:00 | 967.20 | 969.55 | 970.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 13:15:00 | 966.00 | 969.55 | 970.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 13:45:00 | 966.40 | 969.40 | 970.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 15:00:00 | 963.40 | 968.20 | 970.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 09:45:00 | 965.80 | 966.45 | 968.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 960.60 | 962.47 | 965.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 14:15:00 | 959.90 | 961.98 | 964.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 11:30:00 | 959.05 | 958.90 | 961.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 12:00:00 | 958.65 | 958.90 | 961.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 14:30:00 | 960.00 | 959.76 | 961.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 09:15:00 | 975.70 | 962.84 | 962.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2025-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 09:15:00 | 975.70 | 962.84 | 962.46 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 09:15:00 | 960.20 | 963.74 | 964.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 957.05 | 962.40 | 963.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 966.50 | 961.46 | 962.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 14:15:00 | 966.50 | 961.46 | 962.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 966.50 | 961.46 | 962.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 966.50 | 961.46 | 962.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 964.00 | 961.96 | 962.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 973.85 | 961.96 | 962.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 973.05 | 964.18 | 963.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 13:15:00 | 977.65 | 971.37 | 967.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 977.85 | 981.14 | 976.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 10:00:00 | 977.85 | 981.14 | 976.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 981.30 | 981.17 | 977.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:30:00 | 977.25 | 981.17 | 977.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 975.80 | 980.01 | 977.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 13:15:00 | 972.30 | 980.01 | 977.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 978.45 | 979.70 | 977.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 15:00:00 | 978.80 | 979.52 | 977.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 09:45:00 | 980.35 | 979.75 | 978.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 13:15:00 | 982.00 | 982.93 | 983.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 13:15:00 | 982.00 | 982.93 | 983.02 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 14:15:00 | 987.60 | 983.86 | 983.44 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 978.00 | 983.10 | 983.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 12:15:00 | 977.45 | 980.73 | 982.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 09:15:00 | 985.50 | 980.20 | 981.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 09:15:00 | 985.50 | 980.20 | 981.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 985.50 | 980.20 | 981.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:00:00 | 985.50 | 980.20 | 981.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 10:15:00 | 990.10 | 982.18 | 982.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 11:15:00 | 1008.00 | 987.35 | 984.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 11:15:00 | 998.00 | 999.65 | 993.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-22 11:30:00 | 997.30 | 999.65 | 993.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 1000.50 | 999.06 | 994.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 10:30:00 | 1002.20 | 999.91 | 996.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 11:15:00 | 995.10 | 1002.86 | 1003.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 11:15:00 | 995.10 | 1002.86 | 1003.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 13:15:00 | 993.45 | 999.84 | 1001.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 10:15:00 | 981.75 | 980.68 | 987.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 11:00:00 | 981.75 | 980.68 | 987.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 983.55 | 981.26 | 987.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 15:15:00 | 978.70 | 984.88 | 987.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 14:15:00 | 929.76 | 939.42 | 948.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 13:15:00 | 934.15 | 931.13 | 939.34 | SL hit (close>ema200) qty=0.50 sl=931.13 alert=retest2 |

### Cycle 77 — BUY (started 2025-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 11:15:00 | 949.00 | 939.25 | 938.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 12:15:00 | 950.70 | 941.54 | 939.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 09:15:00 | 954.00 | 954.09 | 949.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 09:15:00 | 954.00 | 954.09 | 949.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 954.00 | 954.09 | 949.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 13:45:00 | 959.20 | 954.95 | 951.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 15:00:00 | 958.95 | 955.75 | 952.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 947.65 | 951.69 | 951.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 09:15:00 | 947.65 | 951.69 | 951.70 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 957.65 | 952.89 | 952.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 11:15:00 | 960.10 | 954.33 | 952.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 14:15:00 | 955.80 | 956.46 | 954.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 15:00:00 | 955.80 | 956.46 | 954.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 15:15:00 | 955.00 | 956.17 | 954.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 09:15:00 | 961.45 | 956.17 | 954.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 976.50 | 985.84 | 986.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 976.50 | 985.84 | 986.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 971.75 | 980.95 | 983.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 09:15:00 | 970.85 | 970.55 | 975.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 09:45:00 | 972.10 | 970.55 | 975.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 982.20 | 972.88 | 976.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 982.20 | 972.88 | 976.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 988.20 | 975.94 | 977.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:30:00 | 987.00 | 975.94 | 977.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2025-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 13:15:00 | 990.45 | 980.29 | 978.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 09:15:00 | 1010.95 | 989.23 | 983.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 1017.05 | 1018.27 | 1006.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 14:00:00 | 1017.05 | 1018.27 | 1006.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 1009.50 | 1016.59 | 1008.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 11:45:00 | 1020.00 | 1015.03 | 1009.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 14:30:00 | 1020.80 | 1017.03 | 1011.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 14:15:00 | 1000.50 | 1012.43 | 1012.23 | SL hit (close<static) qty=1.00 sl=1006.20 alert=retest2 |

### Cycle 82 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 1009.00 | 1011.74 | 1011.94 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 09:15:00 | 1021.50 | 1013.69 | 1012.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 12:15:00 | 1036.00 | 1024.38 | 1019.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 09:15:00 | 1026.75 | 1028.37 | 1023.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 10:00:00 | 1026.75 | 1028.37 | 1023.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 1023.15 | 1026.81 | 1024.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 15:00:00 | 1023.15 | 1026.81 | 1024.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 1022.50 | 1025.95 | 1024.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 1028.05 | 1025.95 | 1024.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 15:15:00 | 1034.00 | 1036.40 | 1036.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-09-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 15:15:00 | 1034.00 | 1036.40 | 1036.65 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 1048.00 | 1038.72 | 1037.68 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 10:15:00 | 1034.40 | 1037.34 | 1037.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 15:15:00 | 1029.00 | 1034.61 | 1036.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 12:15:00 | 1035.00 | 1032.17 | 1034.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 12:15:00 | 1035.00 | 1032.17 | 1034.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 1035.00 | 1032.17 | 1034.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:00:00 | 1035.00 | 1032.17 | 1034.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 1032.80 | 1032.30 | 1034.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:45:00 | 1035.05 | 1032.30 | 1034.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 1029.70 | 1031.65 | 1033.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 13:45:00 | 1027.35 | 1031.93 | 1033.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 15:00:00 | 1021.10 | 1029.76 | 1032.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 975.98 | 985.62 | 995.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 970.04 | 985.62 | 995.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-26 15:15:00 | 979.00 | 978.60 | 986.69 | SL hit (close>ema200) qty=0.50 sl=978.60 alert=retest2 |

### Cycle 87 — BUY (started 2025-09-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 15:15:00 | 991.05 | 987.50 | 987.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 994.20 | 989.63 | 988.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 993.00 | 995.64 | 993.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 10:15:00 | 993.00 | 995.64 | 993.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 993.00 | 995.64 | 993.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:00:00 | 993.00 | 995.64 | 993.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 993.20 | 995.15 | 993.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 12:30:00 | 996.15 | 997.49 | 994.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-09 11:15:00 | 994.20 | 1000.67 | 1001.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 11:15:00 | 994.20 | 1000.67 | 1001.29 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2025-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 14:15:00 | 1005.95 | 1001.06 | 1000.65 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 993.30 | 1000.35 | 1000.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 10:15:00 | 992.15 | 998.71 | 999.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 14:15:00 | 998.00 | 995.93 | 997.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 14:15:00 | 998.00 | 995.93 | 997.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 998.00 | 995.93 | 997.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 15:00:00 | 998.00 | 995.93 | 997.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 998.15 | 996.37 | 997.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:15:00 | 1000.50 | 996.37 | 997.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 991.95 | 995.49 | 997.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 10:30:00 | 991.25 | 993.71 | 996.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 10:15:00 | 1000.60 | 997.26 | 996.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 1000.60 | 997.26 | 996.88 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 14:15:00 | 994.60 | 996.42 | 996.61 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 15:15:00 | 997.60 | 996.48 | 996.40 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 09:15:00 | 993.80 | 995.94 | 996.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 10:15:00 | 991.40 | 995.03 | 995.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-17 12:15:00 | 995.25 | 994.56 | 995.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 12:15:00 | 995.25 | 994.56 | 995.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 995.25 | 994.56 | 995.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:45:00 | 994.95 | 994.56 | 995.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 995.95 | 994.84 | 995.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 14:00:00 | 995.95 | 994.84 | 995.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 995.00 | 994.87 | 995.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 15:00:00 | 995.00 | 994.87 | 995.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 993.20 | 994.54 | 995.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:15:00 | 992.55 | 994.54 | 995.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 988.60 | 993.35 | 994.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 10:15:00 | 986.60 | 993.35 | 994.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 13:15:00 | 987.05 | 991.14 | 993.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 14:45:00 | 987.15 | 989.74 | 992.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 10:15:00 | 996.85 | 993.79 | 993.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 10:15:00 | 996.85 | 993.79 | 993.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 15:15:00 | 1000.00 | 996.11 | 994.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 09:15:00 | 994.70 | 995.83 | 994.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 09:15:00 | 994.70 | 995.83 | 994.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 994.70 | 995.83 | 994.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 12:45:00 | 997.95 | 995.74 | 994.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 13:15:00 | 997.20 | 995.74 | 994.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 14:00:00 | 997.50 | 996.09 | 995.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 09:15:00 | 1000.05 | 996.31 | 995.41 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 1000.60 | 997.16 | 995.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 09:15:00 | 1009.00 | 999.32 | 997.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 12:30:00 | 1005.90 | 1001.48 | 999.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 1007.35 | 1002.70 | 1000.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 10:00:00 | 1008.65 | 1003.89 | 1001.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 1009.80 | 1012.39 | 1009.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:00:00 | 1009.80 | 1012.39 | 1009.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 1002.45 | 1010.40 | 1008.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 1002.45 | 1010.40 | 1008.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 1003.55 | 1009.03 | 1008.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:30:00 | 1001.15 | 1009.03 | 1008.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 1002.80 | 1009.22 | 1008.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 15:00:00 | 1002.80 | 1009.22 | 1008.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 1006.00 | 1008.57 | 1008.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:15:00 | 1004.60 | 1008.57 | 1008.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 1007.20 | 1008.38 | 1008.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:45:00 | 1007.60 | 1008.38 | 1008.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-03 11:15:00 | 1003.00 | 1007.31 | 1007.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 11:15:00 | 1003.00 | 1007.31 | 1007.89 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-11-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 15:15:00 | 1012.00 | 1008.42 | 1008.17 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 09:15:00 | 1001.20 | 1006.98 | 1007.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 10:15:00 | 1000.40 | 1005.66 | 1006.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 12:15:00 | 997.80 | 991.89 | 997.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 12:15:00 | 997.80 | 991.89 | 997.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 997.80 | 991.89 | 997.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 13:00:00 | 997.80 | 991.89 | 997.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 992.20 | 991.95 | 996.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 14:15:00 | 989.90 | 991.95 | 996.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 15:00:00 | 986.20 | 990.80 | 995.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 10:15:00 | 965.40 | 960.41 | 960.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 10:15:00 | 965.40 | 960.41 | 960.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 13:15:00 | 968.30 | 963.12 | 961.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 956.80 | 963.43 | 962.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 956.80 | 963.43 | 962.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 956.80 | 963.43 | 962.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:00:00 | 956.80 | 963.43 | 962.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 955.80 | 961.91 | 961.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:30:00 | 954.80 | 961.91 | 961.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 11:15:00 | 953.50 | 960.23 | 960.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 952.40 | 957.48 | 959.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 14:15:00 | 954.90 | 950.04 | 953.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 14:15:00 | 954.90 | 950.04 | 953.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 954.90 | 950.04 | 953.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 954.90 | 950.04 | 953.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 956.70 | 951.37 | 954.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 944.00 | 951.37 | 954.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 10:00:00 | 952.30 | 949.81 | 949.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 10:15:00 | 952.00 | 950.25 | 950.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 10:15:00 | 952.00 | 950.25 | 950.15 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 12:15:00 | 943.30 | 949.09 | 949.65 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 15:15:00 | 952.00 | 949.14 | 948.76 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 12:15:00 | 946.95 | 949.90 | 950.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 10:15:00 | 945.00 | 947.50 | 948.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 14:15:00 | 946.00 | 941.71 | 943.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 14:15:00 | 946.00 | 941.71 | 943.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 946.00 | 941.71 | 943.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 15:00:00 | 946.00 | 941.71 | 943.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 945.95 | 942.56 | 943.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 940.65 | 942.56 | 943.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 934.60 | 931.22 | 934.96 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 942.80 | 936.67 | 936.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 12:15:00 | 950.40 | 940.03 | 938.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 09:15:00 | 935.00 | 941.68 | 939.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 935.00 | 941.68 | 939.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 935.00 | 941.68 | 939.89 | EMA400 retest candle locked (from upside) |

### Cycle 106 — SELL (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 12:15:00 | 934.00 | 938.30 | 938.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 15:15:00 | 930.05 | 935.78 | 937.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 11:15:00 | 935.00 | 931.00 | 933.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 11:15:00 | 935.00 | 931.00 | 933.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 935.00 | 931.00 | 933.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:45:00 | 935.15 | 931.00 | 933.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 935.35 | 931.87 | 933.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 12:45:00 | 935.40 | 931.87 | 933.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 926.75 | 932.14 | 933.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 11:15:00 | 926.00 | 932.14 | 933.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 11:45:00 | 926.00 | 930.98 | 932.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 12:15:00 | 926.00 | 930.98 | 932.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 920.45 | 925.32 | 927.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 925.40 | 921.63 | 923.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 15:00:00 | 925.40 | 921.63 | 923.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 925.50 | 922.41 | 924.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 927.00 | 922.41 | 924.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 924.30 | 923.53 | 924.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:15:00 | 923.05 | 923.53 | 924.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 12:00:00 | 922.00 | 923.22 | 924.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 09:15:00 | 930.80 | 925.44 | 924.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 930.80 | 925.44 | 924.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 11:15:00 | 934.40 | 929.55 | 927.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 14:15:00 | 930.80 | 930.89 | 928.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 15:00:00 | 930.80 | 930.89 | 928.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 931.50 | 931.39 | 929.55 | EMA400 retest candle locked (from upside) |

### Cycle 108 — SELL (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 14:15:00 | 925.75 | 928.81 | 928.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 15:15:00 | 924.40 | 927.93 | 928.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 902.80 | 900.77 | 907.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 10:15:00 | 906.65 | 901.94 | 907.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 906.65 | 901.94 | 907.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 906.65 | 901.94 | 907.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 909.00 | 904.05 | 907.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:30:00 | 909.00 | 904.05 | 907.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 909.20 | 905.08 | 907.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 14:00:00 | 909.20 | 905.08 | 907.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 909.30 | 905.93 | 907.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 15:00:00 | 909.30 | 905.93 | 907.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 906.55 | 906.68 | 907.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 12:00:00 | 906.55 | 906.68 | 907.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 906.10 | 906.56 | 907.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 12:45:00 | 907.15 | 906.56 | 907.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2026-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 13:15:00 | 928.00 | 910.85 | 909.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 10:15:00 | 932.75 | 921.13 | 915.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 926.85 | 932.69 | 926.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:00:00 | 926.85 | 932.69 | 926.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 924.30 | 931.01 | 926.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 15:00:00 | 924.30 | 931.01 | 926.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 930.80 | 930.97 | 926.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:15:00 | 924.20 | 930.97 | 926.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 927.00 | 930.17 | 926.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:30:00 | 925.90 | 930.17 | 926.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 925.30 | 929.20 | 926.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:45:00 | 926.15 | 929.20 | 926.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 921.65 | 927.69 | 925.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 12:00:00 | 921.65 | 927.69 | 925.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 924.75 | 927.10 | 925.87 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2026-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 14:15:00 | 919.50 | 924.83 | 925.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 10:15:00 | 915.95 | 921.77 | 923.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 14:15:00 | 919.95 | 919.19 | 921.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 14:45:00 | 919.75 | 919.19 | 921.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 921.95 | 919.74 | 921.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:15:00 | 920.80 | 919.74 | 921.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 926.80 | 921.15 | 921.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:45:00 | 927.15 | 921.15 | 921.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 921.00 | 921.12 | 921.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:15:00 | 919.65 | 921.12 | 921.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:45:00 | 919.25 | 920.50 | 921.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 14:45:00 | 919.25 | 918.59 | 920.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 10:15:00 | 873.67 | 881.16 | 889.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 10:15:00 | 873.29 | 881.16 | 889.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 10:15:00 | 873.29 | 881.16 | 889.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 879.35 | 876.06 | 882.45 | SL hit (close>ema200) qty=0.50 sl=876.06 alert=retest2 |

### Cycle 111 — BUY (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 11:15:00 | 844.60 | 835.19 | 834.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 09:15:00 | 851.45 | 840.41 | 837.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 837.90 | 841.28 | 838.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 837.90 | 841.28 | 838.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 837.90 | 841.28 | 838.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 837.90 | 841.28 | 838.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 837.65 | 840.55 | 838.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 841.10 | 840.55 | 838.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 828.65 | 838.17 | 837.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 828.65 | 838.17 | 837.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 829.00 | 836.34 | 836.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 15:15:00 | 826.95 | 834.46 | 835.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 832.70 | 822.48 | 826.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 832.70 | 822.48 | 826.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 832.70 | 822.48 | 826.65 | EMA400 retest candle locked (from downside) |

### Cycle 113 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 847.65 | 830.79 | 829.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 857.65 | 840.23 | 835.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 862.70 | 871.56 | 863.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 09:15:00 | 862.70 | 871.56 | 863.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 862.70 | 871.56 | 863.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 09:15:00 | 877.05 | 866.07 | 863.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 11:15:00 | 867.90 | 889.84 | 892.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 11:15:00 | 867.90 | 889.84 | 892.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 12:15:00 | 864.85 | 884.84 | 889.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 843.50 | 839.84 | 849.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 10:00:00 | 843.50 | 839.84 | 849.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 839.80 | 841.84 | 846.01 | EMA400 retest candle locked (from downside) |

### Cycle 115 — BUY (started 2026-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 15:15:00 | 854.95 | 848.65 | 847.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 11:15:00 | 857.25 | 850.95 | 849.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-20 15:15:00 | 853.00 | 854.01 | 851.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 15:15:00 | 853.00 | 854.01 | 851.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 853.00 | 854.01 | 851.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 861.00 | 854.01 | 851.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 10:15:00 | 856.45 | 858.98 | 856.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 13:00:00 | 856.55 | 858.29 | 857.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 13:45:00 | 856.95 | 858.08 | 857.03 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 859.00 | 858.27 | 857.21 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-25 12:15:00 | 849.20 | 855.80 | 856.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2026-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 12:15:00 | 849.20 | 855.80 | 856.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 844.75 | 852.13 | 853.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 12:15:00 | 853.15 | 851.16 | 852.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 12:15:00 | 853.15 | 851.16 | 852.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 853.15 | 851.16 | 852.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 13:00:00 | 853.15 | 851.16 | 852.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 855.85 | 852.10 | 852.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 14:00:00 | 855.85 | 852.10 | 852.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 842.80 | 850.24 | 852.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 14:45:00 | 858.50 | 850.24 | 852.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 807.80 | 806.48 | 810.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 14:30:00 | 811.00 | 806.48 | 810.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 15:15:00 | 790.00 | 788.78 | 793.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:15:00 | 800.45 | 788.78 | 793.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 802.00 | 791.42 | 794.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:45:00 | 802.50 | 791.42 | 794.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 799.35 | 793.01 | 794.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 10:30:00 | 799.00 | 793.01 | 794.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 793.35 | 794.23 | 795.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:30:00 | 795.75 | 794.23 | 795.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 792.95 | 793.98 | 794.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 14:30:00 | 793.70 | 793.98 | 794.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 802.15 | 794.31 | 794.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:00:00 | 802.15 | 794.31 | 794.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 794.25 | 794.30 | 794.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 14:45:00 | 790.90 | 793.49 | 794.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:30:00 | 784.70 | 790.19 | 792.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 785.25 | 778.50 | 778.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 785.25 | 778.50 | 778.41 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 775.60 | 779.20 | 779.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 773.25 | 778.01 | 779.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 09:15:00 | 743.75 | 734.47 | 742.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 09:15:00 | 743.75 | 734.47 | 742.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 743.75 | 734.47 | 742.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 743.75 | 734.47 | 742.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 740.60 | 735.69 | 742.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 11:30:00 | 733.25 | 735.55 | 741.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 12:15:00 | 734.15 | 735.55 | 741.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 09:15:00 | 696.59 | 703.88 | 717.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 09:15:00 | 697.44 | 703.88 | 717.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 708.70 | 688.98 | 700.90 | SL hit (close>ema200) qty=0.50 sl=688.98 alert=retest2 |

### Cycle 119 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 712.05 | 706.29 | 706.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 15:15:00 | 717.20 | 708.47 | 707.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 698.00 | 706.38 | 706.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 698.00 | 706.38 | 706.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 698.00 | 706.38 | 706.32 | EMA400 retest candle locked (from upside) |

### Cycle 120 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 702.75 | 705.65 | 706.00 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 11:15:00 | 710.10 | 706.54 | 706.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 12:15:00 | 716.85 | 708.60 | 707.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 09:15:00 | 710.00 | 715.62 | 711.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-06 09:15:00 | 710.00 | 715.62 | 711.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 710.00 | 715.62 | 711.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 10:00:00 | 710.00 | 715.62 | 711.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 707.75 | 714.04 | 711.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 11:00:00 | 707.75 | 714.04 | 711.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 717.75 | 717.47 | 714.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 09:45:00 | 717.05 | 717.47 | 714.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 740.00 | 743.42 | 738.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 748.00 | 743.42 | 738.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 15:00:00 | 742.55 | 745.32 | 742.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 735.00 | 742.80 | 741.49 | SL hit (close<static) qty=1.00 sl=738.00 alert=retest2 |

### Cycle 122 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 879.30 | 886.75 | 887.01 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2026-05-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 15:15:00 | 889.00 | 883.48 | 883.36 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 10:15:00 | 881.55 | 883.02 | 883.17 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2026-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 15:15:00 | 887.90 | 883.57 | 883.26 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 12:15:00 | 874.00 | 881.45 | 882.39 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 09:15:00 | 890.25 | 882.73 | 882.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 10:15:00 | 901.00 | 886.39 | 884.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 14:15:00 | 894.25 | 894.64 | 889.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 15:00:00 | 894.25 | 894.64 | 889.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 895.05 | 895.60 | 891.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:00:00 | 895.05 | 895.60 | 891.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 888.30 | 894.14 | 891.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:45:00 | 887.10 | 894.14 | 891.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 886.85 | 892.68 | 890.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:45:00 | 886.75 | 892.68 | 890.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 890.85 | 891.15 | 890.35 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-06-03 10:45:00 | 1193.05 | 2024-06-04 09:15:00 | 1133.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 12:45:00 | 1194.60 | 2024-06-04 09:15:00 | 1134.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 15:15:00 | 1194.00 | 2024-06-04 09:15:00 | 1134.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 10:45:00 | 1193.05 | 2024-06-04 10:15:00 | 1073.74 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-03 12:45:00 | 1194.60 | 2024-06-04 10:15:00 | 1075.14 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-03 15:15:00 | 1194.00 | 2024-06-04 10:15:00 | 1074.60 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-06-28 11:45:00 | 1690.00 | 2024-07-09 09:15:00 | 1676.95 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2024-07-12 15:00:00 | 1616.80 | 2024-07-19 14:15:00 | 1535.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-15 09:15:00 | 1593.20 | 2024-07-22 09:15:00 | 1513.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-12 15:00:00 | 1616.80 | 2024-07-23 09:15:00 | 1538.70 | STOP_HIT | 0.50 | 4.83% |
| SELL | retest2 | 2024-07-15 09:15:00 | 1593.20 | 2024-07-23 09:15:00 | 1538.70 | STOP_HIT | 0.50 | 3.42% |
| SELL | retest2 | 2024-08-05 09:15:00 | 1539.15 | 2024-08-06 15:15:00 | 1462.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-05 09:15:00 | 1539.15 | 2024-08-07 12:15:00 | 1495.70 | STOP_HIT | 0.50 | 2.82% |
| SELL | retest2 | 2024-09-04 09:15:00 | 1506.80 | 2024-09-05 09:15:00 | 1431.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-04 10:30:00 | 1505.60 | 2024-09-05 09:15:00 | 1430.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-04 09:15:00 | 1506.80 | 2024-09-09 14:15:00 | 1412.00 | STOP_HIT | 0.50 | 6.29% |
| SELL | retest2 | 2024-09-04 10:30:00 | 1505.60 | 2024-09-09 14:15:00 | 1412.00 | STOP_HIT | 0.50 | 6.22% |
| SELL | retest2 | 2024-09-26 11:15:00 | 1345.15 | 2024-09-27 12:15:00 | 1391.85 | STOP_HIT | 1.00 | -3.47% |
| SELL | retest2 | 2024-09-27 10:45:00 | 1344.05 | 2024-09-27 12:15:00 | 1391.85 | STOP_HIT | 1.00 | -3.56% |
| BUY | retest2 | 2024-09-30 12:30:00 | 1392.10 | 2024-10-03 11:15:00 | 1359.05 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2024-10-07 10:15:00 | 1331.10 | 2024-10-09 11:15:00 | 1382.30 | STOP_HIT | 1.00 | -3.85% |
| SELL | retest2 | 2024-10-08 09:30:00 | 1327.00 | 2024-10-09 11:15:00 | 1382.30 | STOP_HIT | 1.00 | -4.17% |
| SELL | retest2 | 2024-10-08 10:00:00 | 1324.10 | 2024-10-09 11:15:00 | 1382.30 | STOP_HIT | 1.00 | -4.40% |
| BUY | retest2 | 2024-10-15 09:15:00 | 1399.40 | 2024-10-15 09:15:00 | 1386.75 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-10-15 10:30:00 | 1395.25 | 2024-10-17 09:15:00 | 1372.90 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2024-10-15 11:30:00 | 1395.00 | 2024-10-17 09:15:00 | 1372.90 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2024-10-15 15:00:00 | 1393.40 | 2024-10-17 09:15:00 | 1372.90 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2024-11-04 14:00:00 | 1373.40 | 2024-11-05 09:15:00 | 1326.80 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2024-11-12 11:15:00 | 1280.35 | 2024-11-19 12:15:00 | 1271.05 | STOP_HIT | 1.00 | 0.73% |
| BUY | retest2 | 2024-12-02 15:15:00 | 1295.00 | 2024-12-10 10:15:00 | 1312.60 | STOP_HIT | 1.00 | 1.36% |
| SELL | retest2 | 2024-12-12 09:15:00 | 1308.55 | 2024-12-18 12:15:00 | 1243.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-12 09:15:00 | 1308.55 | 2024-12-23 12:15:00 | 1177.69 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-14 12:15:00 | 1052.00 | 2025-01-16 09:15:00 | 1074.60 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-01-15 10:00:00 | 1051.50 | 2025-01-16 09:15:00 | 1074.60 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2025-01-15 13:15:00 | 1051.00 | 2025-01-16 09:15:00 | 1074.60 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2025-01-15 15:15:00 | 1052.00 | 2025-01-16 09:15:00 | 1074.60 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2025-01-17 14:15:00 | 1077.15 | 2025-01-22 10:15:00 | 1065.15 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-01-28 14:45:00 | 1029.05 | 2025-01-29 09:15:00 | 1063.50 | STOP_HIT | 1.00 | -3.35% |
| SELL | retest2 | 2025-01-29 12:30:00 | 1032.85 | 2025-02-01 09:15:00 | 1087.00 | STOP_HIT | 1.00 | -5.24% |
| SELL | retest2 | 2025-01-29 13:00:00 | 1031.90 | 2025-02-01 09:15:00 | 1087.00 | STOP_HIT | 1.00 | -5.34% |
| SELL | retest2 | 2025-01-29 13:45:00 | 1032.00 | 2025-02-01 09:15:00 | 1087.00 | STOP_HIT | 1.00 | -5.33% |
| SELL | retest2 | 2025-01-30 14:15:00 | 1029.35 | 2025-02-01 09:15:00 | 1087.00 | STOP_HIT | 1.00 | -5.60% |
| SELL | retest2 | 2025-01-31 14:00:00 | 1023.25 | 2025-02-01 09:15:00 | 1087.00 | STOP_HIT | 1.00 | -6.23% |
| SELL | retest2 | 2025-02-18 11:00:00 | 957.00 | 2025-02-19 14:15:00 | 974.30 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-02-19 10:00:00 | 954.25 | 2025-02-19 14:15:00 | 974.30 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-02-24 13:00:00 | 1017.20 | 2025-02-27 13:15:00 | 1002.85 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-02-25 09:15:00 | 1025.15 | 2025-02-27 13:15:00 | 1002.85 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2025-02-27 09:45:00 | 1018.00 | 2025-02-27 13:15:00 | 1002.85 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-02-27 12:30:00 | 1017.80 | 2025-02-27 13:15:00 | 1002.85 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-03-04 13:15:00 | 961.85 | 2025-03-05 09:15:00 | 997.85 | STOP_HIT | 1.00 | -3.74% |
| SELL | retest2 | 2025-03-04 14:30:00 | 962.90 | 2025-03-05 09:15:00 | 997.85 | STOP_HIT | 1.00 | -3.63% |
| SELL | retest2 | 2025-03-04 15:15:00 | 962.20 | 2025-03-05 09:15:00 | 997.85 | STOP_HIT | 1.00 | -3.71% |
| BUY | retest2 | 2025-03-07 11:15:00 | 1013.30 | 2025-03-11 09:15:00 | 984.90 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2025-03-10 10:00:00 | 1019.40 | 2025-03-11 09:15:00 | 984.90 | STOP_HIT | 1.00 | -3.38% |
| BUY | retest2 | 2025-03-10 11:15:00 | 1015.80 | 2025-03-11 09:15:00 | 984.90 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2025-03-10 12:45:00 | 1017.55 | 2025-03-11 09:15:00 | 984.90 | STOP_HIT | 1.00 | -3.21% |
| SELL | retest2 | 2025-03-13 12:45:00 | 975.00 | 2025-03-18 09:15:00 | 991.60 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-03-17 11:00:00 | 974.95 | 2025-03-18 09:15:00 | 991.60 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-03-17 14:15:00 | 973.75 | 2025-03-18 09:15:00 | 991.60 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-03-21 10:15:00 | 1053.65 | 2025-03-26 14:15:00 | 1040.00 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-04-04 10:15:00 | 980.75 | 2025-04-07 09:15:00 | 882.68 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-04 13:00:00 | 983.55 | 2025-04-07 09:15:00 | 885.19 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-07 09:15:00 | 936.25 | 2025-04-07 09:15:00 | 842.62 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-11 09:45:00 | 979.95 | 2025-04-11 10:15:00 | 981.50 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-04-28 09:15:00 | 978.90 | 2025-04-29 09:15:00 | 996.30 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest1 | 2025-04-28 10:15:00 | 983.70 | 2025-04-29 09:15:00 | 996.30 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest1 | 2025-04-28 14:15:00 | 982.00 | 2025-04-29 09:15:00 | 996.30 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-04-29 14:15:00 | 988.30 | 2025-05-06 14:15:00 | 938.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-30 09:15:00 | 978.50 | 2025-05-06 14:15:00 | 929.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-29 14:15:00 | 988.30 | 2025-05-08 09:15:00 | 962.65 | STOP_HIT | 0.50 | 2.60% |
| SELL | retest2 | 2025-04-30 09:15:00 | 978.50 | 2025-05-08 09:15:00 | 962.65 | STOP_HIT | 0.50 | 1.62% |
| SELL | retest2 | 2025-05-22 13:45:00 | 1008.85 | 2025-05-23 11:15:00 | 1027.05 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-05-28 10:45:00 | 1049.65 | 2025-05-30 12:15:00 | 1037.90 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-05-28 11:30:00 | 1056.50 | 2025-05-30 12:15:00 | 1037.90 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-06-16 09:15:00 | 993.00 | 2025-06-27 10:15:00 | 977.20 | STOP_HIT | 1.00 | 1.59% |
| SELL | retest2 | 2025-06-16 11:15:00 | 1000.50 | 2025-06-27 10:15:00 | 977.20 | STOP_HIT | 1.00 | 2.33% |
| SELL | retest2 | 2025-06-16 11:45:00 | 1001.00 | 2025-06-27 10:15:00 | 977.20 | STOP_HIT | 1.00 | 2.38% |
| SELL | retest2 | 2025-06-16 12:15:00 | 1001.20 | 2025-06-27 10:15:00 | 977.20 | STOP_HIT | 1.00 | 2.40% |
| SELL | retest2 | 2025-06-17 09:15:00 | 998.00 | 2025-06-27 10:15:00 | 977.20 | STOP_HIT | 1.00 | 2.08% |
| SELL | retest2 | 2025-06-30 13:15:00 | 966.00 | 2025-07-04 09:15:00 | 975.70 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-06-30 13:45:00 | 966.40 | 2025-07-04 09:15:00 | 975.70 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-06-30 15:00:00 | 963.40 | 2025-07-04 09:15:00 | 975.70 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-07-01 09:45:00 | 965.80 | 2025-07-04 09:15:00 | 975.70 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-07-02 14:15:00 | 959.90 | 2025-07-04 09:15:00 | 975.70 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-07-03 11:30:00 | 959.05 | 2025-07-04 09:15:00 | 975.70 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-07-03 12:00:00 | 958.65 | 2025-07-04 09:15:00 | 975.70 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-07-03 14:30:00 | 960.00 | 2025-07-04 09:15:00 | 975.70 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-07-11 15:00:00 | 978.80 | 2025-07-16 13:15:00 | 982.00 | STOP_HIT | 1.00 | 0.33% |
| BUY | retest2 | 2025-07-14 09:45:00 | 980.35 | 2025-07-16 13:15:00 | 982.00 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest2 | 2025-07-23 10:30:00 | 1002.20 | 2025-07-25 11:15:00 | 995.10 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-07-30 15:15:00 | 978.70 | 2025-08-06 14:15:00 | 929.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 15:15:00 | 978.70 | 2025-08-07 13:15:00 | 934.15 | STOP_HIT | 0.50 | 4.55% |
| BUY | retest2 | 2025-08-13 13:45:00 | 959.20 | 2025-08-18 09:15:00 | 947.65 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-08-13 15:00:00 | 958.95 | 2025-08-18 09:15:00 | 947.65 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-08-19 09:15:00 | 961.45 | 2025-08-26 09:15:00 | 976.50 | STOP_HIT | 1.00 | 1.57% |
| BUY | retest2 | 2025-09-03 11:45:00 | 1020.00 | 2025-09-04 14:15:00 | 1000.50 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-09-03 14:30:00 | 1020.80 | 2025-09-04 14:15:00 | 1000.50 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2025-09-10 09:15:00 | 1028.05 | 2025-09-15 15:15:00 | 1034.00 | STOP_HIT | 1.00 | 0.58% |
| SELL | retest2 | 2025-09-19 13:45:00 | 1027.35 | 2025-09-26 09:15:00 | 975.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 15:00:00 | 1021.10 | 2025-09-26 09:15:00 | 970.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 13:45:00 | 1027.35 | 2025-09-26 15:15:00 | 979.00 | STOP_HIT | 0.50 | 4.71% |
| SELL | retest2 | 2025-09-19 15:00:00 | 1021.10 | 2025-09-26 15:15:00 | 979.00 | STOP_HIT | 0.50 | 4.12% |
| BUY | retest2 | 2025-10-06 12:30:00 | 996.15 | 2025-10-09 11:15:00 | 994.20 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-10-14 10:30:00 | 991.25 | 2025-10-15 10:15:00 | 1000.60 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-10-20 10:15:00 | 986.60 | 2025-10-23 10:15:00 | 996.85 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-10-20 13:15:00 | 987.05 | 2025-10-23 10:15:00 | 996.85 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-10-20 14:45:00 | 987.15 | 2025-10-23 10:15:00 | 996.85 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-10-24 12:45:00 | 997.95 | 2025-11-03 11:15:00 | 1003.00 | STOP_HIT | 1.00 | 0.51% |
| BUY | retest2 | 2025-10-24 13:15:00 | 997.20 | 2025-11-03 11:15:00 | 1003.00 | STOP_HIT | 1.00 | 0.58% |
| BUY | retest2 | 2025-10-24 14:00:00 | 997.50 | 2025-11-03 11:15:00 | 1003.00 | STOP_HIT | 1.00 | 0.55% |
| BUY | retest2 | 2025-10-27 09:15:00 | 1000.05 | 2025-11-03 11:15:00 | 1003.00 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2025-10-28 09:15:00 | 1009.00 | 2025-11-03 11:15:00 | 1003.00 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-10-28 12:30:00 | 1005.90 | 2025-11-03 11:15:00 | 1003.00 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2025-10-29 09:15:00 | 1007.35 | 2025-11-03 11:15:00 | 1003.00 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-10-29 10:00:00 | 1008.65 | 2025-11-03 11:15:00 | 1003.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-11-06 14:15:00 | 989.90 | 2025-11-20 10:15:00 | 965.40 | STOP_HIT | 1.00 | 2.47% |
| SELL | retest2 | 2025-11-06 15:00:00 | 986.20 | 2025-11-20 10:15:00 | 965.40 | STOP_HIT | 1.00 | 2.11% |
| SELL | retest2 | 2025-11-25 09:15:00 | 944.00 | 2025-11-27 10:15:00 | 952.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-11-27 10:00:00 | 952.30 | 2025-11-27 10:15:00 | 952.00 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2025-12-16 11:15:00 | 926.00 | 2025-12-22 09:15:00 | 930.80 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-12-16 11:45:00 | 926.00 | 2025-12-22 09:15:00 | 930.80 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-12-16 12:15:00 | 926.00 | 2025-12-22 09:15:00 | 930.80 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-12-18 09:15:00 | 920.45 | 2025-12-22 09:15:00 | 930.80 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-12-19 11:15:00 | 923.05 | 2025-12-22 09:15:00 | 930.80 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-12-19 12:00:00 | 922.00 | 2025-12-22 09:15:00 | 930.80 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-01-08 11:15:00 | 919.65 | 2026-01-14 10:15:00 | 873.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 11:45:00 | 919.25 | 2026-01-14 10:15:00 | 873.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 14:45:00 | 919.25 | 2026-01-14 10:15:00 | 873.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 11:15:00 | 919.65 | 2026-01-16 09:15:00 | 879.35 | STOP_HIT | 0.50 | 4.38% |
| SELL | retest2 | 2026-01-08 11:45:00 | 919.25 | 2026-01-16 09:15:00 | 879.35 | STOP_HIT | 0.50 | 4.34% |
| SELL | retest2 | 2026-01-08 14:45:00 | 919.25 | 2026-01-16 09:15:00 | 879.35 | STOP_HIT | 0.50 | 4.34% |
| BUY | retest2 | 2026-02-09 09:15:00 | 877.05 | 2026-02-12 11:15:00 | 867.90 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2026-02-23 09:15:00 | 861.00 | 2026-02-25 12:15:00 | 849.20 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2026-02-24 10:15:00 | 856.45 | 2026-02-25 12:15:00 | 849.20 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2026-02-24 13:00:00 | 856.55 | 2026-02-25 12:15:00 | 849.20 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-02-24 13:45:00 | 856.95 | 2026-02-25 12:15:00 | 849.20 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2026-03-12 14:45:00 | 790.90 | 2026-03-18 09:15:00 | 785.25 | STOP_HIT | 1.00 | 0.71% |
| SELL | retest2 | 2026-03-13 09:30:00 | 784.70 | 2026-03-18 09:15:00 | 785.25 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2026-03-25 11:30:00 | 733.25 | 2026-03-30 09:15:00 | 696.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 12:15:00 | 734.15 | 2026-03-30 09:15:00 | 697.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 11:30:00 | 733.25 | 2026-04-01 09:15:00 | 708.70 | STOP_HIT | 0.50 | 3.35% |
| SELL | retest2 | 2026-03-25 12:15:00 | 734.15 | 2026-04-01 09:15:00 | 708.70 | STOP_HIT | 0.50 | 3.47% |
| BUY | retest2 | 2026-04-10 09:15:00 | 748.00 | 2026-04-13 09:15:00 | 735.00 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2026-04-10 15:00:00 | 742.55 | 2026-04-13 09:15:00 | 735.00 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2026-04-13 10:45:00 | 747.85 | 2026-04-22 09:15:00 | 816.15 | TARGET_HIT | 1.00 | 9.13% |
| BUY | retest2 | 2026-04-13 13:30:00 | 741.95 | 2026-04-22 10:15:00 | 822.64 | TARGET_HIT | 1.00 | 10.87% |
| BUY | retest2 | 2026-04-15 09:15:00 | 755.75 | 2026-04-22 10:15:00 | 831.33 | TARGET_HIT | 1.00 | 10.00% |
