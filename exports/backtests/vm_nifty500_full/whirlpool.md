# Whirlpool of India Ltd. (WHIRLPOOL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 985.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT3 | 6 |
| ENTRY1 | 6 |
| ENTRY2 | 2 |
| EXIT | 6 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 6 / 2
- **Target hits / EMA400 exits:** 5 / 3
- **Total realized P&L (per unit):** 693.74
- **Avg P&L per closed trade:** 86.72

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 09:15:00 | 1566.10 | 1605.35 | 1605.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-24 10:15:00 | 1558.25 | 1604.88 | 1605.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 12:15:00 | 1390.05 | 1382.22 | 1432.55 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-01-19 10:15:00 | 1367.45 | 1381.98 | 1431.18 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-04-03 11:15:00 | 1321.10 | 1265.10 | 1302.51 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-04-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 11:15:00 | 1424.90 | 1329.27 | 1328.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-19 14:15:00 | 1439.80 | 1338.24 | 1333.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-07 11:15:00 | 1414.45 | 1417.21 | 1382.76 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-07 13:15:00 | 1438.45 | 1417.45 | 1383.22 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-06-04 11:15:00 | 1425.70 | 1477.63 | 1437.78 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 09:15:00 | 1749.75 | 2127.29 | 2128.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 10:15:00 | 1741.95 | 2123.45 | 2126.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 14:15:00 | 1931.05 | 1929.52 | 1995.70 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-11 09:15:00 | 1899.00 | 1929.21 | 1994.89 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 1976.95 | 1929.41 | 1980.31 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-20 13:15:00 | 1934.85 | 1930.11 | 1979.90 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 10:15:00 | 1138.90 | 1062.73 | 1148.49 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-24 13:15:00 | 1161.50 | 1065.10 | 1148.41 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 13:15:00 | 1281.00 | 1191.98 | 1191.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 10:15:00 | 1303.80 | 1196.06 | 1193.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 1220.20 | 1221.28 | 1209.04 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-30 10:15:00 | 1237.70 | 1221.45 | 1209.19 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 1343.10 | 1379.06 | 1340.30 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-31 13:15:00 | 1340.00 | 1378.01 | 1340.35 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 14:15:00 | 1306.90 | 1319.59 | 1319.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 15:15:00 | 1303.10 | 1319.43 | 1319.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 14:15:00 | 1314.30 | 1314.19 | 1316.67 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 11:15:00 | 1366.10 | 1319.13 | 1319.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 09:15:00 | 1373.00 | 1321.01 | 1319.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 14:15:00 | 1328.50 | 1332.85 | 1326.55 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-12 09:15:00 | 1338.50 | 1332.86 | 1326.62 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 1338.50 | 1332.86 | 1326.62 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-09-12 10:15:00 | 1354.00 | 1333.07 | 1326.76 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 1331.30 | 1337.26 | 1329.90 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-18 13:15:00 | 1326.70 | 1337.15 | 1329.88 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 14:15:00 | 1233.50 | 1323.99 | 1324.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 10:15:00 | 1222.10 | 1321.14 | 1322.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 14:15:00 | 1239.60 | 1238.11 | 1271.85 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 10:15:00 | 1407.30 | 1297.68 | 1297.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 14:15:00 | 1423.00 | 1302.09 | 1299.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 09:15:00 | 1314.00 | 1320.46 | 1309.89 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 10:15:00 | 1252.20 | 1301.49 | 1301.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 13:15:00 | 1242.70 | 1299.83 | 1300.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 13:15:00 | 852.15 | 841.76 | 928.57 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-13 09:15:00 | 839.00 | 887.72 | 913.17 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 858.15 | 838.03 | 865.35 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-17 12:15:00 | 874.00 | 838.76 | 865.31 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 10:15:00 | 989.15 | 885.34 | 885.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 11:15:00 | 993.00 | 886.41 | 885.68 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-01-19 10:15:00 | 1367.45 | 2024-04-03 11:15:00 | 1321.10 | EXIT_EMA400 | 46.35 |
| BUY | 2024-05-07 13:15:00 | 1438.45 | 2024-05-21 09:15:00 | 1604.14 | TARGET | 165.69 |
| SELL | 2024-12-20 13:15:00 | 1934.85 | 2024-12-30 14:15:00 | 1799.70 | TARGET | 135.15 |
| SELL | 2024-12-11 09:15:00 | 1899.00 | 2025-01-13 12:15:00 | 1611.32 | TARGET | 287.68 |
| BUY | 2025-05-30 10:15:00 | 1237.70 | 2025-06-09 09:15:00 | 1323.24 | TARGET | 85.54 |
| BUY | 2025-09-12 09:15:00 | 1338.50 | 2025-09-16 12:15:00 | 1374.13 | TARGET | 35.63 |
| BUY | 2025-09-12 10:15:00 | 1354.00 | 2025-09-18 13:15:00 | 1326.70 | EXIT_EMA400 | -27.30 |
| SELL | 2026-03-13 09:15:00 | 839.00 | 2026-04-17 12:15:00 | 874.00 | EXIT_EMA400 | -35.00 |
