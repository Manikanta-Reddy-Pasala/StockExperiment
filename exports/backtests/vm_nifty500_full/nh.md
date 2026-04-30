# Narayana Hrudayalaya Ltd. (NH.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 1766.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 6 |
| ENTRY1 | 7 |
| ENTRY2 | 4 |
| EXIT | 7 |

## P&L

- **Trades closed:** 11
- **Trades open at end:** 0
- **Winners / losers:** 3 / 8
- **Target hits / EMA400 exits:** 3 / 8
- **Total realized P&L (per unit):** -14.02
- **Avg P&L per closed trade:** -1.27

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-10 12:15:00 | 1264.10 | 1271.94 | 1271.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-10 13:15:00 | 1258.00 | 1271.81 | 1271.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-17 09:15:00 | 1276.95 | 1268.05 | 1269.88 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-22 14:15:00 | 1265.80 | 1271.34 | 1271.46 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-05-23 09:15:00 | 1272.80 | 1271.30 | 1271.44 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 13:15:00 | 1270.15 | 1242.16 | 1242.15 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-08-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 13:15:00 | 1221.45 | 1242.26 | 1242.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 14:15:00 | 1217.60 | 1242.02 | 1242.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-09 11:15:00 | 1246.00 | 1239.70 | 1240.97 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-09 14:15:00 | 1230.60 | 1239.59 | 1240.90 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 14:15:00 | 1230.60 | 1239.59 | 1240.90 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-08-12 09:15:00 | 1221.85 | 1239.35 | 1240.76 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-08-21 10:15:00 | 1232.90 | 1222.53 | 1231.25 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 14:15:00 | 1273.55 | 1238.23 | 1238.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 09:15:00 | 1317.95 | 1250.36 | 1244.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 09:15:00 | 1284.55 | 1284.92 | 1266.04 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2024-10-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 12:15:00 | 1206.20 | 1255.67 | 1255.86 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-10-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 13:15:00 | 1288.60 | 1255.07 | 1254.91 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2024-10-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 11:15:00 | 1231.90 | 1255.02 | 1255.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 13:15:00 | 1222.20 | 1254.45 | 1254.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-28 09:15:00 | 1255.90 | 1254.11 | 1254.56 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-29 09:15:00 | 1216.50 | 1253.09 | 1254.03 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 10:15:00 | 1251.45 | 1249.92 | 1252.31 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-10-31 11:15:00 | 1257.60 | 1249.99 | 1252.34 | Close above EMA400 |

### Cycle 8 — BUY (started 2024-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 12:15:00 | 1292.30 | 1253.48 | 1253.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 11:15:00 | 1318.10 | 1259.93 | 1257.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-11 11:15:00 | 1273.00 | 1276.36 | 1266.79 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-12 10:15:00 | 1297.80 | 1276.83 | 1267.30 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 1285.65 | 1281.69 | 1271.43 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-12-19 10:15:00 | 1288.80 | 1281.76 | 1271.52 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 1279.50 | 1282.65 | 1272.63 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-12-23 10:15:00 | 1308.00 | 1282.91 | 1272.80 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-12-30 14:15:00 | 1274.05 | 1288.24 | 1277.21 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 13:15:00 | 1817.50 | 1869.55 | 1869.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 10:15:00 | 1803.60 | 1865.55 | 1867.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-04 13:15:00 | 1836.00 | 1835.13 | 1850.19 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-05 09:15:00 | 1815.40 | 1834.94 | 1849.87 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 1806.70 | 1807.75 | 1830.61 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-17 12:15:00 | 1800.00 | 1807.70 | 1830.36 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 1802.70 | 1774.80 | 1803.35 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-06 10:15:00 | 1833.00 | 1775.37 | 1803.50 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 13:15:00 | 1953.50 | 1796.40 | 1795.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 09:15:00 | 1996.60 | 1801.41 | 1798.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 12:15:00 | 1893.80 | 1893.96 | 1857.99 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-08 14:15:00 | 1923.40 | 1894.41 | 1858.57 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-12-10 13:15:00 | 1857.20 | 1893.47 | 1860.37 | Close below EMA400 |

### Cycle 11 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 1723.50 | 1866.94 | 1866.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 12:15:00 | 1711.20 | 1863.94 | 1865.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-10 12:15:00 | 1806.50 | 1805.44 | 1830.72 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-10 14:15:00 | 1795.60 | 1805.30 | 1830.40 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-11 09:15:00 | 1837.20 | 1805.51 | 1830.25 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-05-22 14:15:00 | 1265.80 | 2024-05-23 09:15:00 | 1272.80 | EXIT_EMA400 | -7.00 |
| SELL | 2024-08-09 14:15:00 | 1230.60 | 2024-08-13 10:15:00 | 1199.71 | TARGET | 30.89 |
| SELL | 2024-08-12 09:15:00 | 1221.85 | 2024-08-21 10:15:00 | 1232.90 | EXIT_EMA400 | -11.05 |
| SELL | 2024-10-29 09:15:00 | 1216.50 | 2024-10-31 11:15:00 | 1257.60 | EXIT_EMA400 | -41.10 |
| BUY | 2024-12-12 10:15:00 | 1297.80 | 2024-12-30 14:15:00 | 1274.05 | EXIT_EMA400 | -23.75 |
| BUY | 2024-12-19 10:15:00 | 1288.80 | 2024-12-30 14:15:00 | 1274.05 | EXIT_EMA400 | -14.75 |
| BUY | 2024-12-23 10:15:00 | 1308.00 | 2024-12-30 14:15:00 | 1274.05 | EXIT_EMA400 | -33.95 |
| SELL | 2025-09-05 09:15:00 | 1815.40 | 2025-09-26 09:15:00 | 1711.98 | TARGET | 103.42 |
| SELL | 2025-09-17 12:15:00 | 1800.00 | 2025-09-26 09:15:00 | 1708.93 | TARGET | 91.07 |
| BUY | 2025-12-08 14:15:00 | 1923.40 | 2025-12-10 13:15:00 | 1857.20 | EXIT_EMA400 | -66.20 |
| SELL | 2026-02-10 14:15:00 | 1795.60 | 2026-02-11 09:15:00 | 1837.20 | EXIT_EMA400 | -41.60 |
