# TECHM (TECHM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:30:00 (4997 bars)
- **Last close:** 1466.70
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 4 |
| ALERT3 | 11 |
| PENDING | 36 |
| PENDING_CANCEL | 8 |
| ENTRY1 | 6 |
| ENTRY2 | 21 |
| PARTIAL | 5 |
| TARGET_HIT | 1 |
| STOP_HIT | 26 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 32 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 23
- **Target hits / Stop hits / Partials:** 1 / 26 / 5
- **Avg / median % per leg:** 2.19% / -1.56%
- **Sum % (uncompounded):** 70.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 7 | 77.8% | 1 | 4 | 4 | 10.59% | 95.3% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 0 | 3 | 2 | 6.42% | 32.1% |
| BUY @ 3rd Alert (retest2) | 4 | 4 | 100.0% | 1 | 1 | 2 | 15.80% | 63.2% |
| SELL (all) | 23 | 2 | 8.7% | 0 | 22 | 1 | -1.10% | -25.3% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.53% | -10.6% |
| SELL @ 3rd Alert (retest2) | 20 | 2 | 10.0% | 0 | 19 | 1 | -0.74% | -14.7% |
| retest1 (combined) | 8 | 3 | 37.5% | 0 | 6 | 2 | 2.69% | 21.5% |
| retest2 (combined) | 24 | 6 | 25.0% | 1 | 20 | 3 | 2.02% | 48.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-26 11:15:00 | 1108.45 | 1206.81 | 1206.86 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-04 12:15:00 | 1219.80 | 1190.02 | 1189.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 14:15:00 | 1224.00 | 1190.66 | 1190.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-13 11:15:00 | 1199.00 | 1203.45 | 1197.45 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-13 11:15:00 | 1199.00 | 1203.45 | 1197.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 11:15:00 | 1199.00 | 1203.45 | 1197.45 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2023-12-13 14:15:00 | 1216.20 | 1203.61 | 1197.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-13 15:15:00 | 1216.10 | 1203.73 | 1197.71 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-01-15 09:15:00 | 1398.51 | 1251.56 | 1233.93 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-27 11:15:00 | 1255.25 | 1282.56 | 1282.65 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |

### Cycle 3 — SELL (started 2024-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-27 11:15:00 | 1255.25 | 1282.56 | 1282.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-27 15:15:00 | 1247.00 | 1281.34 | 1282.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-04 13:15:00 | 1274.00 | 1272.74 | 1277.25 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-04 14:15:00 | 1281.55 | 1272.83 | 1277.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 14:15:00 | 1281.55 | 1272.83 | 1277.27 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-04-05 09:15:00 | 1269.00 | 1272.85 | 1277.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-05 10:15:00 | 1266.45 | 1272.79 | 1277.19 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-04-26 09:15:00 | 1289.00 | 1239.08 | 1256.11 | SL hit qty=1.00 sl=1289.00 alert=retest2 |
| Cross detected — sustain check pending | 2024-04-30 10:15:00 | 1265.30 | 1246.04 | 1258.49 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-30 11:15:00 | 1268.00 | 1246.25 | 1258.54 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-05-07 14:15:00 | 1289.00 | 1250.81 | 1259.21 | SL hit qty=1.00 sl=1289.00 alert=retest2 |
| Cross detected — sustain check pending | 2024-05-09 09:15:00 | 1268.55 | 1254.08 | 1260.52 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 10:15:00 | 1269.95 | 1254.24 | 1260.57 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-05-09 12:15:00 | 1271.60 | 1254.65 | 1260.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 13:15:00 | 1269.25 | 1254.80 | 1260.76 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 14:15:00 | 1267.35 | 1254.92 | 1260.79 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-05-10 10:15:00 | 1265.05 | 1255.31 | 1260.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-10 11:15:00 | 1251.00 | 1255.27 | 1260.85 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-05-14 13:15:00 | 1273.95 | 1256.22 | 1260.91 | SL hit qty=1.00 sl=1273.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-16 09:15:00 | 1289.00 | 1258.18 | 1261.68 | SL hit qty=1.00 sl=1289.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-16 09:15:00 | 1289.00 | 1258.18 | 1261.68 | SL hit qty=1.00 sl=1289.00 alert=retest2 |

### Cycle 4 — BUY (started 2024-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-21 10:15:00 | 1314.00 | 1264.99 | 1264.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-21 11:15:00 | 1315.90 | 1265.49 | 1265.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-30 09:15:00 | 1276.00 | 1286.37 | 1277.09 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 09:15:00 | 1276.00 | 1286.37 | 1277.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 1276.00 | 1286.37 | 1277.09 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-06-06 09:15:00 | 1294.05 | 1275.82 | 1272.80 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-06 10:15:00 | 1292.10 | 1275.99 | 1272.90 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-07-02 09:15:00 | 1485.91 | 1363.36 | 1327.93 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Target hit — 30% from entry | 2024-10-14 11:15:00 | 1679.73 | 1616.61 | 1579.61 | Target hit (30%) qty=0.50 alert=retest2 |

### Cycle 5 — SELL (started 2025-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 14:15:00 | 1651.60 | 1690.74 | 1690.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 1633.45 | 1687.53 | 1689.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 12:15:00 | 1682.70 | 1680.20 | 1684.93 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 12:15:00 | 1682.70 | 1680.20 | 1684.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 12:15:00 | 1682.70 | 1680.20 | 1684.93 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-02-11 12:15:00 | 1664.60 | 1681.12 | 1685.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-02-11 13:15:00 | 1675.45 | 1681.07 | 1685.04 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-02-11 14:15:00 | 1669.90 | 1680.96 | 1684.97 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 15:15:00 | 1669.10 | 1680.84 | 1684.89 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-02-12 09:15:00 | 1688.00 | 1680.86 | 1684.88 | SL hit qty=1.00 sl=1688.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-02-14 10:15:00 | 1661.15 | 1680.59 | 1684.45 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 11:15:00 | 1666.05 | 1680.44 | 1684.36 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-02-18 09:15:00 | 1688.00 | 1677.96 | 1682.85 | SL hit qty=1.00 sl=1688.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-02-20 11:15:00 | 1670.80 | 1679.91 | 1683.49 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-20 12:15:00 | 1663.15 | 1679.74 | 1683.39 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-03-12 11:15:00 | 1413.68 | 1586.25 | 1627.43 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-20 14:15:00 | 1575.50 | 1503.54 | 1503.44 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |

### Cycle 6 — BUY (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 14:15:00 | 1575.50 | 1503.54 | 1503.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 09:15:00 | 1590.00 | 1505.14 | 1504.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 09:15:00 | 1536.80 | 1540.67 | 1525.25 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-06-04 13:15:00 | 1565.00 | 1541.83 | 1526.67 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-04 14:15:00 | 1556.80 | 1541.98 | 1526.82 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-06-05 11:15:00 | 1568.20 | 1542.67 | 1527.47 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-05 12:15:00 | 1561.90 | 1542.86 | 1527.64 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-06-06 09:15:00 | 1565.50 | 1543.65 | 1528.34 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 10:15:00 | 1571.50 | 1543.93 | 1528.55 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1599.80 | 1639.80 | 1604.60 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 1604.60 | 1639.80 | 1604.60 | SL hit qty=1.00 sl=1604.60 alert=retest1 |

### Cycle 7 — SELL (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 10:15:00 | 1460.30 | 1585.46 | 1585.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 13:15:00 | 1447.50 | 1581.53 | 1583.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 10:15:00 | 1525.00 | 1520.49 | 1544.79 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-08-14 11:15:00 | 1502.90 | 1520.31 | 1544.58 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 12:15:00 | 1495.10 | 1520.06 | 1544.33 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-22 10:15:00 | 1499.80 | 1515.12 | 1537.86 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-22 11:15:00 | 1507.70 | 1515.05 | 1537.71 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-22 15:15:00 | 1503.00 | 1514.76 | 1537.12 | ENTRY1 cross detected — sustain check pending (15m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 1527.90 | 1514.89 | 1537.07 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-08-25 09:15:00 | 1537.07 | 1514.89 | 1537.07 | SL hit qty=1.00 sl=1537.07 alert=retest1 |
| Cross detected — sustain check pending | 2025-08-26 13:15:00 | 1512.30 | 1515.82 | 1536.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 14:15:00 | 1500.90 | 1515.67 | 1536.19 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-02 10:15:00 | 1514.20 | 1512.10 | 1531.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-02 11:15:00 | 1515.60 | 1512.13 | 1531.87 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-02 13:15:00 | 1512.80 | 1512.21 | 1531.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 14:15:00 | 1513.40 | 1512.22 | 1531.62 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-03 09:15:00 | 1508.70 | 1512.22 | 1531.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 10:15:00 | 1503.00 | 1512.13 | 1531.29 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-11 09:15:00 | 1507.70 | 1506.74 | 1524.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 10:15:00 | 1513.50 | 1506.80 | 1524.64 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 1521.60 | 1507.47 | 1524.46 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-09-15 09:15:00 | 1510.80 | 1508.50 | 1524.40 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 10:15:00 | 1511.40 | 1508.53 | 1524.33 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-16 14:15:00 | 1527.60 | 1509.86 | 1524.17 | SL hit qty=1.00 sl=1527.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-17 09:15:00 | 1539.60 | 1510.25 | 1524.22 | SL hit qty=1.00 sl=1539.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-17 09:15:00 | 1539.60 | 1510.25 | 1524.22 | SL hit qty=1.00 sl=1539.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-17 09:15:00 | 1539.60 | 1510.25 | 1524.22 | SL hit qty=1.00 sl=1539.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-17 09:15:00 | 1539.60 | 1510.25 | 1524.22 | SL hit qty=1.00 sl=1539.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-22 09:15:00 | 1498.10 | 1516.87 | 1526.31 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 10:15:00 | 1497.10 | 1516.67 | 1526.17 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-11-27 09:15:00 | 1527.60 | 1455.15 | 1464.63 | SL hit qty=1.00 sl=1527.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-27 12:15:00 | 1506.00 | 1456.90 | 1465.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 13:15:00 | 1506.30 | 1457.39 | 1465.57 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-01 10:15:00 | 1512.70 | 1463.61 | 1468.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-01 11:15:00 | 1520.00 | 1464.17 | 1468.58 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-12-01 13:15:00 | 1527.60 | 1465.38 | 1469.15 | SL hit qty=1.00 sl=1527.60 alert=retest2 |

### Cycle 8 — BUY (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 11:15:00 | 1545.90 | 1472.95 | 1472.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 09:15:00 | 1574.70 | 1476.68 | 1474.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 14:15:00 | 1578.30 | 1579.95 | 1546.96 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-01-09 09:15:00 | 1588.60 | 1580.02 | 1547.32 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-09 10:15:00 | 1591.40 | 1580.13 | 1547.54 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-09 14:15:00 | 1584.70 | 1580.28 | 1548.26 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-09 15:15:00 | 1582.20 | 1580.30 | 1548.43 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-12 11:15:00 | 1583.50 | 1580.06 | 1548.78 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-12 12:15:00 | 1585.70 | 1580.11 | 1548.97 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-02-03 09:15:00 | 1830.11 | 1656.59 | 1605.80 | Partial book 0.50 @ 15%; trail SL->entry alert=retest1 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-02-03 09:15:00 | 1823.55 | 1656.59 | 1605.80 | Partial book 0.50 @ 15%; trail SL->entry alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 1628.90 | 1660.54 | 1609.57 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 1591.40 | 1650.36 | 1613.60 | SL hit qty=0.50 sl=1591.40 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 1585.70 | 1650.36 | 1613.60 | SL hit qty=0.50 sl=1585.70 alert=retest1 |

### Cycle 9 — SELL (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 14:15:00 | 1439.60 | 1586.36 | 1586.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 09:15:00 | 1394.00 | 1583.02 | 1584.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 1422.90 | 1415.56 | 1471.97 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-03-25 15:15:00 | 1404.00 | 1416.27 | 1469.32 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 09:15:00 | 1407.50 | 1416.18 | 1469.01 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 2520m) |
| Cross detected — sustain check pending | 2026-04-01 13:15:00 | 1406.30 | 1412.77 | 1462.67 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-01 14:15:00 | 1404.40 | 1412.69 | 1462.38 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 1459.40 | 1413.44 | 1460.57 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-04-06 09:15:00 | 1460.57 | 1413.44 | 1460.57 | SL hit qty=1.00 sl=1460.57 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-06 09:15:00 | 1460.57 | 1413.44 | 1460.57 | SL hit qty=1.00 sl=1460.57 alert=retest1 |
| Cross detected — sustain check pending | 2026-04-09 09:15:00 | 1437.50 | 1421.99 | 1460.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-09 10:15:00 | 1444.90 | 1422.22 | 1460.30 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-10 09:15:00 | 1436.60 | 1424.08 | 1460.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 10:15:00 | 1431.00 | 1424.15 | 1459.97 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-10 15:15:00 | 1438.00 | 1424.74 | 1459.39 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 1423.30 | 1424.72 | 1459.21 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 3960m) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 1464.80 | 1425.65 | 1458.49 | SL hit qty=1.00 sl=1464.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 1464.80 | 1425.65 | 1458.49 | SL hit qty=1.00 sl=1464.80 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-22 10:15:00 | 1422.60 | 1443.78 | 1463.41 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 11:15:00 | 1410.50 | 1443.45 | 1463.14 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-22 13:15:00 | 1464.80 | 1443.45 | 1462.95 | SL hit qty=1.00 sl=1464.80 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-23 09:15:00 | 1425.00 | 1443.65 | 1462.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 10:15:00 | 1434.60 | 1443.56 | 1462.62 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 1458.60 | 1432.87 | 1454.37 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 1464.80 | 1434.37 | 1454.50 | SL hit qty=1.00 sl=1464.80 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-12-13 15:15:00 | 1216.10 | 2024-01-15 09:15:00 | 1398.51 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2023-12-13 15:15:00 | 1216.10 | 2024-03-27 11:15:00 | 1255.25 | STOP_HIT | 0.50 | 3.22% |
| SELL | retest2 | 2024-04-05 10:15:00 | 1266.45 | 2024-04-26 09:15:00 | 1289.00 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-04-30 11:15:00 | 1268.00 | 2024-05-07 14:15:00 | 1289.00 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2024-05-09 10:15:00 | 1269.95 | 2024-05-14 13:15:00 | 1273.95 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2024-05-09 13:15:00 | 1269.25 | 2024-05-16 09:15:00 | 1289.00 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2024-05-10 11:15:00 | 1251.00 | 2024-05-16 09:15:00 | 1289.00 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2024-06-06 10:15:00 | 1292.10 | 2024-07-02 09:15:00 | 1485.91 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2024-06-06 10:15:00 | 1292.10 | 2024-10-14 11:15:00 | 1679.73 | TARGET_HIT | 0.50 | 30.00% |
| SELL | retest2 | 2025-02-11 15:15:00 | 1669.10 | 2025-02-12 09:15:00 | 1688.00 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-02-14 11:15:00 | 1666.05 | 2025-02-18 09:15:00 | 1688.00 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-02-20 12:15:00 | 1663.15 | 2025-03-12 11:15:00 | 1413.68 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2025-02-20 12:15:00 | 1663.15 | 2025-05-20 14:15:00 | 1575.50 | STOP_HIT | 0.50 | 5.27% |
| BUY | retest1 | 2025-06-06 10:15:00 | 1571.50 | 2025-07-10 09:15:00 | 1604.60 | STOP_HIT | 1.00 | 2.11% |
| SELL | retest1 | 2025-08-14 12:15:00 | 1495.10 | 2025-08-25 09:15:00 | 1537.07 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2025-08-26 14:15:00 | 1500.90 | 2025-09-16 14:15:00 | 1527.60 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-09-02 14:15:00 | 1513.40 | 2025-09-17 09:15:00 | 1539.60 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-09-03 10:15:00 | 1503.00 | 2025-09-17 09:15:00 | 1539.60 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-09-11 10:15:00 | 1513.50 | 2025-09-17 09:15:00 | 1539.60 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-09-15 10:15:00 | 1511.40 | 2025-09-17 09:15:00 | 1539.60 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-09-22 10:15:00 | 1497.10 | 2025-11-27 09:15:00 | 1527.60 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-11-27 13:15:00 | 1506.30 | 2025-12-01 13:15:00 | 1527.60 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest1 | 2026-01-09 10:15:00 | 1591.40 | 2026-02-03 09:15:00 | 1830.11 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2026-01-12 12:15:00 | 1585.70 | 2026-02-03 09:15:00 | 1823.55 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2026-01-09 10:15:00 | 1591.40 | 2026-02-12 09:15:00 | 1591.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-12 12:15:00 | 1585.70 | 2026-02-12 09:15:00 | 1585.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-27 09:15:00 | 1407.50 | 2026-04-06 09:15:00 | 1460.57 | STOP_HIT | 1.00 | -3.77% |
| SELL | retest1 | 2026-04-01 14:15:00 | 1404.40 | 2026-04-06 09:15:00 | 1460.57 | STOP_HIT | 1.00 | -4.00% |
| SELL | retest2 | 2026-04-10 10:15:00 | 1431.00 | 2026-04-15 09:15:00 | 1464.80 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2026-04-13 09:15:00 | 1423.30 | 2026-04-15 09:15:00 | 1464.80 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2026-04-22 11:15:00 | 1410.50 | 2026-04-22 13:15:00 | 1464.80 | STOP_HIT | 1.00 | -3.85% |
| SELL | retest2 | 2026-04-23 10:15:00 | 1434.60 | 2026-04-30 09:15:00 | 1464.80 | STOP_HIT | 1.00 | -2.11% |
