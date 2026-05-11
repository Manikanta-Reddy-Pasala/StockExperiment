# Acutaas Chemicals Ltd. (ACUTAAS)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (15450 bars)
- **Last close:** 2748.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 0 |
| ALERT1 | 0 |
| ALERT2 | 0 |
| ALERT2_SKIP | 0 |
| ALERT3 | 0 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 38 |
| ENTRY2 | 0 |
| PARTIAL | 16 |
| TARGET_HIT | 7 |
| STOP_HIT | 31 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 54 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 23 / 31
- **Target hits / Stop hits / Partials:** 7 / 31 / 16
- **Avg / median % per leg:** 0.32% / 0.00%
- **Sum % (uncompounded):** 17.48%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 30 | 13 | 43.3% | 4 | 17 | 9 | 0.43% | 13.0% |
| BUY @ 2nd Alert (retest1) | 30 | 13 | 43.3% | 4 | 17 | 9 | 0.43% | 13.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 24 | 10 | 41.7% | 3 | 14 | 7 | 0.19% | 4.5% |
| SELL @ 2nd Alert (retest1) | 24 | 10 | 41.7% | 3 | 14 | 7 | 0.19% | 4.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 54 | 23 | 42.6% | 7 | 31 | 16 | 0.32% | 17.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 10:50:00 | 1188.00 | 1175.89 | 0.00 | ORB-long ORB[1165.20,1178.90] vol=1.8x ATR=5.08 |
| Stop hit — per-position SL triggered | 2025-05-28 11:40:00 | 1182.92 | 1177.89 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-06-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-04 09:30:00 | 1129.70 | 1127.03 | 0.00 | ORB-long ORB[1120.10,1128.90] vol=2.6x ATR=4.19 |
| Stop hit — per-position SL triggered | 2025-06-04 09:50:00 | 1125.51 | 1127.25 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-06-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 10:50:00 | 1151.80 | 1141.41 | 0.00 | ORB-long ORB[1126.00,1142.90] vol=1.8x ATR=6.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-06 11:40:00 | 1160.82 | 1145.47 | 0.00 | T1 1.5R @ 1160.82 |
| Stop hit — per-position SL triggered | 2025-06-06 12:10:00 | 1151.80 | 1147.42 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-06-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-10 09:35:00 | 1109.40 | 1114.57 | 0.00 | ORB-short ORB[1111.00,1124.50] vol=1.6x ATR=4.89 |
| Stop hit — per-position SL triggered | 2025-06-10 10:05:00 | 1114.29 | 1111.78 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-06-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-11 10:55:00 | 1118.80 | 1125.18 | 0.00 | ORB-short ORB[1120.00,1131.70] vol=1.6x ATR=3.41 |
| Stop hit — per-position SL triggered | 2025-06-11 11:20:00 | 1122.21 | 1124.56 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-06-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-26 09:30:00 | 1071.90 | 1076.72 | 0.00 | ORB-short ORB[1075.80,1085.40] vol=2.9x ATR=3.35 |
| Stop hit — per-position SL triggered | 2025-06-26 09:35:00 | 1075.25 | 1075.78 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 09:35:00 | 1121.00 | 1114.26 | 0.00 | ORB-long ORB[1100.10,1115.00] vol=2.5x ATR=5.59 |
| Stop hit — per-position SL triggered | 2025-06-27 09:45:00 | 1115.41 | 1115.80 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-07-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 10:00:00 | 1099.80 | 1111.13 | 0.00 | ORB-short ORB[1112.40,1128.40] vol=2.2x ATR=4.08 |
| Stop hit — per-position SL triggered | 2025-07-02 11:15:00 | 1103.88 | 1106.71 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-07-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 11:00:00 | 1106.80 | 1102.87 | 0.00 | ORB-long ORB[1094.90,1101.80] vol=9.8x ATR=3.24 |
| Stop hit — per-position SL triggered | 2025-07-04 11:05:00 | 1103.56 | 1102.89 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-07-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-07 09:30:00 | 1135.30 | 1130.78 | 0.00 | ORB-long ORB[1122.10,1133.80] vol=1.7x ATR=3.81 |
| Stop hit — per-position SL triggered | 2025-07-07 09:40:00 | 1131.49 | 1131.37 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-07-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 10:55:00 | 1164.00 | 1160.69 | 0.00 | ORB-long ORB[1149.90,1159.50] vol=2.1x ATR=5.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-15 12:35:00 | 1171.51 | 1163.17 | 0.00 | T1 1.5R @ 1171.51 |
| Target hit | 2025-07-15 15:20:00 | 1187.00 | 1173.85 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2025-07-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-18 09:45:00 | 1225.10 | 1218.92 | 0.00 | ORB-long ORB[1210.90,1223.00] vol=1.8x ATR=5.58 |
| Stop hit — per-position SL triggered | 2025-07-18 09:50:00 | 1219.52 | 1220.53 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 09:40:00 | 1190.40 | 1192.30 | 0.00 | ORB-short ORB[1191.80,1199.80] vol=1.7x ATR=3.99 |
| Stop hit — per-position SL triggered | 2025-07-22 09:45:00 | 1194.39 | 1192.26 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-07-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-23 10:30:00 | 1196.20 | 1188.68 | 0.00 | ORB-long ORB[1175.10,1187.80] vol=1.7x ATR=4.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-23 10:35:00 | 1203.24 | 1189.34 | 0.00 | T1 1.5R @ 1203.24 |
| Stop hit — per-position SL triggered | 2025-07-23 11:40:00 | 1196.20 | 1195.28 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-08-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 10:40:00 | 1320.50 | 1312.43 | 0.00 | ORB-long ORB[1305.20,1320.00] vol=1.5x ATR=5.53 |
| Stop hit — per-position SL triggered | 2025-08-13 10:55:00 | 1314.97 | 1312.68 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-08-14 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 11:10:00 | 1297.40 | 1303.70 | 0.00 | ORB-short ORB[1305.30,1319.30] vol=2.9x ATR=2.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-14 13:15:00 | 1293.22 | 1300.90 | 0.00 | T1 1.5R @ 1293.22 |
| Stop hit — per-position SL triggered | 2025-08-14 14:45:00 | 1297.40 | 1297.84 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-09-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 09:50:00 | 1418.50 | 1413.04 | 0.00 | ORB-long ORB[1397.80,1415.00] vol=4.1x ATR=4.90 |
| Stop hit — per-position SL triggered | 2025-09-03 10:00:00 | 1413.60 | 1413.58 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-12 10:15:00 | 1464.10 | 1473.77 | 0.00 | ORB-short ORB[1465.00,1480.00] vol=2.2x ATR=5.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-12 12:25:00 | 1455.89 | 1467.85 | 0.00 | T1 1.5R @ 1455.89 |
| Target hit | 2025-09-12 15:20:00 | 1440.00 | 1454.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — BUY (started 2025-09-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 10:55:00 | 1466.10 | 1460.25 | 0.00 | ORB-long ORB[1452.10,1462.00] vol=1.6x ATR=3.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-17 11:00:00 | 1471.41 | 1461.25 | 0.00 | T1 1.5R @ 1471.41 |
| Stop hit — per-position SL triggered | 2025-09-17 11:50:00 | 1466.10 | 1462.78 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-09-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-22 11:00:00 | 1452.60 | 1455.31 | 0.00 | ORB-short ORB[1452.90,1466.40] vol=1.6x ATR=5.32 |
| Stop hit — per-position SL triggered | 2025-09-22 11:05:00 | 1457.92 | 1455.42 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-09-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 10:00:00 | 1448.80 | 1457.57 | 0.00 | ORB-short ORB[1454.90,1471.40] vol=3.3x ATR=3.65 |
| Stop hit — per-position SL triggered | 2025-09-23 10:05:00 | 1452.45 | 1457.26 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-11-19 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-19 10:10:00 | 1720.10 | 1713.38 | 0.00 | ORB-long ORB[1695.40,1710.60] vol=1.8x ATR=6.08 |
| Stop hit — per-position SL triggered | 2025-11-19 12:05:00 | 1714.02 | 1718.02 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-11-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-20 11:05:00 | 1710.40 | 1725.53 | 0.00 | ORB-short ORB[1724.50,1740.00] vol=1.8x ATR=4.97 |
| Stop hit — per-position SL triggered | 2025-11-20 11:15:00 | 1715.37 | 1725.02 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-11-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 10:35:00 | 1710.60 | 1718.47 | 0.00 | ORB-short ORB[1717.50,1726.00] vol=1.6x ATR=3.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 11:00:00 | 1704.68 | 1716.39 | 0.00 | T1 1.5R @ 1704.68 |
| Stop hit — per-position SL triggered | 2025-11-21 11:05:00 | 1710.60 | 1715.92 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-11-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-24 10:20:00 | 1733.70 | 1713.13 | 0.00 | ORB-long ORB[1694.40,1707.60] vol=1.7x ATR=7.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 11:00:00 | 1745.64 | 1720.47 | 0.00 | T1 1.5R @ 1745.64 |
| Target hit | 2025-11-24 13:10:00 | 1812.00 | 1812.91 | 0.00 | Trail-exit close<VWAP |

### Cycle 26 — SELL (started 2025-11-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 11:00:00 | 1766.30 | 1783.11 | 0.00 | ORB-short ORB[1778.40,1796.10] vol=2.5x ATR=5.27 |
| Stop hit — per-position SL triggered | 2025-11-27 11:25:00 | 1771.57 | 1780.69 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-11-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 09:45:00 | 1808.30 | 1793.18 | 0.00 | ORB-long ORB[1775.00,1791.80] vol=4.8x ATR=7.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-28 09:50:00 | 1819.09 | 1804.37 | 0.00 | T1 1.5R @ 1819.09 |
| Target hit | 2025-11-28 12:10:00 | 1822.30 | 1827.79 | 0.00 | Trail-exit close<VWAP |

### Cycle 28 — SELL (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 11:15:00 | 1811.30 | 1823.39 | 0.00 | ORB-short ORB[1825.10,1848.10] vol=1.9x ATR=6.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-01 12:25:00 | 1801.64 | 1817.54 | 0.00 | T1 1.5R @ 1801.64 |
| Target hit | 2025-12-01 15:20:00 | 1771.60 | 1801.29 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — SELL (started 2025-12-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 09:40:00 | 1695.00 | 1697.72 | 0.00 | ORB-short ORB[1696.50,1718.20] vol=1.8x ATR=6.87 |
| Stop hit — per-position SL triggered | 2025-12-05 10:00:00 | 1701.87 | 1696.54 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-12-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 10:05:00 | 1687.50 | 1695.03 | 0.00 | ORB-short ORB[1692.00,1710.10] vol=2.4x ATR=6.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 10:15:00 | 1678.35 | 1692.81 | 0.00 | T1 1.5R @ 1678.35 |
| Stop hit — per-position SL triggered | 2025-12-08 10:20:00 | 1687.50 | 1692.77 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-15 10:15:00 | 1643.20 | 1648.94 | 0.00 | ORB-short ORB[1644.10,1659.40] vol=1.9x ATR=5.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-15 10:35:00 | 1634.77 | 1645.30 | 0.00 | T1 1.5R @ 1634.77 |
| Target hit | 2025-12-15 15:20:00 | 1630.60 | 1632.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 32 — BUY (started 2025-12-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-22 10:50:00 | 1670.90 | 1665.45 | 0.00 | ORB-long ORB[1645.70,1669.90] vol=2.1x ATR=4.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-22 11:10:00 | 1678.23 | 1667.31 | 0.00 | T1 1.5R @ 1678.23 |
| Stop hit — per-position SL triggered | 2025-12-22 11:15:00 | 1670.90 | 1667.67 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-12-26 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 10:20:00 | 1642.30 | 1653.02 | 0.00 | ORB-short ORB[1652.40,1667.90] vol=1.8x ATR=4.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 10:45:00 | 1635.43 | 1645.49 | 0.00 | T1 1.5R @ 1635.43 |
| Stop hit — per-position SL triggered | 2025-12-26 13:35:00 | 1642.30 | 1635.77 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2026-01-01 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 10:35:00 | 1731.00 | 1713.60 | 0.00 | ORB-long ORB[1695.70,1716.00] vol=3.4x ATR=6.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 10:45:00 | 1741.07 | 1723.71 | 0.00 | T1 1.5R @ 1741.07 |
| Stop hit — per-position SL triggered | 2026-01-01 11:15:00 | 1731.00 | 1725.86 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2026-03-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 10:00:00 | 2114.10 | 2088.72 | 0.00 | ORB-long ORB[2062.00,2091.00] vol=2.1x ATR=9.90 |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 2104.20 | 2094.12 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2026-03-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 09:30:00 | 2205.80 | 2190.01 | 0.00 | ORB-long ORB[2170.00,2198.70] vol=2.1x ATR=14.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 09:40:00 | 2226.93 | 2201.13 | 0.00 | T1 1.5R @ 2226.93 |
| Target hit | 2026-03-17 15:20:00 | 2313.50 | 2278.07 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — BUY (started 2026-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:30:00 | 2311.80 | 2298.23 | 0.00 | ORB-long ORB[2283.00,2306.00] vol=1.7x ATR=10.54 |
| Stop hit — per-position SL triggered | 2026-03-18 09:55:00 | 2301.26 | 2303.72 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2026-05-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 09:30:00 | 2755.00 | 2735.90 | 0.00 | ORB-long ORB[2700.00,2739.00] vol=3.3x ATR=9.81 |
| Stop hit — per-position SL triggered | 2026-05-08 09:35:00 | 2745.19 | 2736.19 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-28 10:50:00 | 1188.00 | 2025-05-28 11:40:00 | 1182.92 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-06-04 09:30:00 | 1129.70 | 2025-06-04 09:50:00 | 1125.51 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-06-06 10:50:00 | 1151.80 | 2025-06-06 11:40:00 | 1160.82 | PARTIAL | 0.50 | 0.78% |
| BUY | retest1 | 2025-06-06 10:50:00 | 1151.80 | 2025-06-06 12:10:00 | 1151.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-10 09:35:00 | 1109.40 | 2025-06-10 10:05:00 | 1114.29 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2025-06-11 10:55:00 | 1118.80 | 2025-06-11 11:20:00 | 1122.21 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-06-26 09:30:00 | 1071.90 | 2025-06-26 09:35:00 | 1075.25 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-06-27 09:35:00 | 1121.00 | 2025-06-27 09:45:00 | 1115.41 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2025-07-02 10:00:00 | 1099.80 | 2025-07-02 11:15:00 | 1103.88 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-07-04 11:00:00 | 1106.80 | 2025-07-04 11:05:00 | 1103.56 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-07-07 09:30:00 | 1135.30 | 2025-07-07 09:40:00 | 1131.49 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-07-15 10:55:00 | 1164.00 | 2025-07-15 12:35:00 | 1171.51 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2025-07-15 10:55:00 | 1164.00 | 2025-07-15 15:20:00 | 1187.00 | TARGET_HIT | 0.50 | 1.98% |
| BUY | retest1 | 2025-07-18 09:45:00 | 1225.10 | 2025-07-18 09:50:00 | 1219.52 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2025-07-22 09:40:00 | 1190.40 | 2025-07-22 09:45:00 | 1194.39 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-07-23 10:30:00 | 1196.20 | 2025-07-23 10:35:00 | 1203.24 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-07-23 10:30:00 | 1196.20 | 2025-07-23 11:40:00 | 1196.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-13 10:40:00 | 1320.50 | 2025-08-13 10:55:00 | 1314.97 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-08-14 11:10:00 | 1297.40 | 2025-08-14 13:15:00 | 1293.22 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-08-14 11:10:00 | 1297.40 | 2025-08-14 14:45:00 | 1297.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-03 09:50:00 | 1418.50 | 2025-09-03 10:00:00 | 1413.60 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-09-12 10:15:00 | 1464.10 | 2025-09-12 12:25:00 | 1455.89 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2025-09-12 10:15:00 | 1464.10 | 2025-09-12 15:20:00 | 1440.00 | TARGET_HIT | 0.50 | 1.65% |
| BUY | retest1 | 2025-09-17 10:55:00 | 1466.10 | 2025-09-17 11:00:00 | 1471.41 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-09-17 10:55:00 | 1466.10 | 2025-09-17 11:50:00 | 1466.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-22 11:00:00 | 1452.60 | 2025-09-22 11:05:00 | 1457.92 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-09-23 10:00:00 | 1448.80 | 2025-09-23 10:05:00 | 1452.45 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-11-19 10:10:00 | 1720.10 | 2025-11-19 12:05:00 | 1714.02 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-11-20 11:05:00 | 1710.40 | 2025-11-20 11:15:00 | 1715.37 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-11-21 10:35:00 | 1710.60 | 2025-11-21 11:00:00 | 1704.68 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-11-21 10:35:00 | 1710.60 | 2025-11-21 11:05:00 | 1710.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-24 10:20:00 | 1733.70 | 2025-11-24 11:00:00 | 1745.64 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2025-11-24 10:20:00 | 1733.70 | 2025-11-24 13:10:00 | 1812.00 | TARGET_HIT | 0.50 | 4.52% |
| SELL | retest1 | 2025-11-27 11:00:00 | 1766.30 | 2025-11-27 11:25:00 | 1771.57 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-11-28 09:45:00 | 1808.30 | 2025-11-28 09:50:00 | 1819.09 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2025-11-28 09:45:00 | 1808.30 | 2025-11-28 12:10:00 | 1822.30 | TARGET_HIT | 0.50 | 0.77% |
| SELL | retest1 | 2025-12-01 11:15:00 | 1811.30 | 2025-12-01 12:25:00 | 1801.64 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-12-01 11:15:00 | 1811.30 | 2025-12-01 15:20:00 | 1771.60 | TARGET_HIT | 0.50 | 2.19% |
| SELL | retest1 | 2025-12-05 09:40:00 | 1695.00 | 2025-12-05 10:00:00 | 1701.87 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-12-08 10:05:00 | 1687.50 | 2025-12-08 10:15:00 | 1678.35 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-12-08 10:05:00 | 1687.50 | 2025-12-08 10:20:00 | 1687.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-15 10:15:00 | 1643.20 | 2025-12-15 10:35:00 | 1634.77 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-12-15 10:15:00 | 1643.20 | 2025-12-15 15:20:00 | 1630.60 | TARGET_HIT | 0.50 | 0.77% |
| BUY | retest1 | 2025-12-22 10:50:00 | 1670.90 | 2025-12-22 11:10:00 | 1678.23 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-12-22 10:50:00 | 1670.90 | 2025-12-22 11:15:00 | 1670.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-26 10:20:00 | 1642.30 | 2025-12-26 10:45:00 | 1635.43 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-12-26 10:20:00 | 1642.30 | 2025-12-26 13:35:00 | 1642.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-01 10:35:00 | 1731.00 | 2026-01-01 10:45:00 | 1741.07 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-01-01 10:35:00 | 1731.00 | 2026-01-01 11:15:00 | 1731.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-16 10:00:00 | 2114.10 | 2026-03-16 10:15:00 | 2104.20 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-03-17 09:30:00 | 2205.80 | 2026-03-17 09:40:00 | 2226.93 | PARTIAL | 0.50 | 0.96% |
| BUY | retest1 | 2026-03-17 09:30:00 | 2205.80 | 2026-03-17 15:20:00 | 2313.50 | TARGET_HIT | 0.50 | 4.88% |
| BUY | retest1 | 2026-03-18 09:30:00 | 2311.80 | 2026-03-18 09:55:00 | 2301.26 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2026-05-08 09:30:00 | 2755.00 | 2026-05-08 09:35:00 | 2745.19 | STOP_HIT | 1.00 | -0.36% |
