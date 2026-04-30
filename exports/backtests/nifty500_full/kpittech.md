# KPIT Technologies Ltd. (KPITTECH.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 759.05
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT3 | 1 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / EMA400 exits:** 3 / 4
- **Total realized P&L (per unit):** 146.52
- **Avg P&L per closed trade:** 20.93

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 14:15:00 | 1356.05 | 1508.02 | 1508.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-20 09:15:00 | 1350.05 | 1504.97 | 1506.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-27 14:15:00 | 1480.05 | 1473.40 | 1489.17 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-12 09:15:00 | 1470.00 | 1488.07 | 1493.33 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-04-29 13:15:00 | 1510.00 | 1445.29 | 1466.77 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-05-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 09:15:00 | 1525.70 | 1479.49 | 1479.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 10:15:00 | 1537.25 | 1480.06 | 1479.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 12:15:00 | 1489.45 | 1491.33 | 1485.78 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 1352.70 | 1480.98 | 1481.06 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-06-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 13:15:00 | 1511.15 | 1480.24 | 1480.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-19 15:15:00 | 1515.10 | 1482.80 | 1481.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 10:15:00 | 1726.40 | 1750.94 | 1669.12 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-12 09:15:00 | 1769.05 | 1745.35 | 1678.83 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-09-05 09:15:00 | 1737.15 | 1791.31 | 1739.08 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 15:15:00 | 1673.85 | 1723.60 | 1723.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 10:15:00 | 1659.70 | 1718.08 | 1720.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-09 09:15:00 | 1727.30 | 1716.95 | 1720.01 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2024-10-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 14:15:00 | 1788.55 | 1723.07 | 1722.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-11 15:15:00 | 1795.50 | 1723.79 | 1723.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 11:15:00 | 1737.30 | 1741.15 | 1732.88 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2024-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 11:15:00 | 1438.00 | 1725.84 | 1725.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 1356.85 | 1710.18 | 1717.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 09:15:00 | 1439.30 | 1437.60 | 1521.67 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-23 12:15:00 | 1410.00 | 1480.84 | 1517.07 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-30 09:15:00 | 1466.55 | 1375.44 | 1428.02 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 11:15:00 | 1321.50 | 1284.78 | 1284.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 09:15:00 | 1342.20 | 1286.63 | 1285.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 09:15:00 | 1324.40 | 1350.09 | 1326.58 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2025-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 12:15:00 | 1270.50 | 1310.34 | 1310.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 1263.60 | 1308.27 | 1309.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-16 10:15:00 | 1295.50 | 1295.27 | 1301.86 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-16 13:15:00 | 1291.60 | 1295.28 | 1301.77 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1226.50 | 1220.95 | 1242.21 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-08 09:15:00 | 1220.20 | 1221.44 | 1241.73 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-09-09 09:15:00 | 1243.00 | 1222.01 | 1241.32 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 14:15:00 | 1259.80 | 1206.29 | 1206.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 09:15:00 | 1283.20 | 1207.57 | 1206.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-09 09:15:00 | 1206.30 | 1216.06 | 1211.50 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-11 11:15:00 | 1228.70 | 1214.93 | 1211.26 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-12-15 13:15:00 | 1211.30 | 1217.11 | 1212.68 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 14:15:00 | 1162.40 | 1208.54 | 1208.66 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 09:15:00 | 1230.50 | 1208.85 | 1208.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-24 09:15:00 | 1234.40 | 1210.43 | 1209.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 14:15:00 | 1207.40 | 1211.75 | 1210.31 | EMA200 retest candle locked |

### Cycle 13 — SELL (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 13:15:00 | 1168.10 | 1208.77 | 1208.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 14:15:00 | 1160.60 | 1208.29 | 1208.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 1193.40 | 1193.28 | 1200.45 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-20 10:15:00 | 1136.90 | 1186.83 | 1195.12 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-04-12 09:15:00 | 1470.00 | 2024-04-18 14:15:00 | 1400.00 | TARGET | 70.00 |
| BUY | 2024-08-12 09:15:00 | 1769.05 | 2024-09-05 09:15:00 | 1737.15 | EXIT_EMA400 | -31.90 |
| SELL | 2024-12-23 12:15:00 | 1410.00 | 2025-01-30 09:15:00 | 1466.55 | EXIT_EMA400 | -56.55 |
| SELL | 2025-07-16 13:15:00 | 1291.60 | 2025-07-21 09:15:00 | 1261.09 | TARGET | 30.51 |
| SELL | 2025-09-08 09:15:00 | 1220.20 | 2025-09-09 09:15:00 | 1243.00 | EXIT_EMA400 | -22.80 |
| BUY | 2025-12-11 11:15:00 | 1228.70 | 2025-12-15 13:15:00 | 1211.30 | EXIT_EMA400 | -17.40 |
| SELL | 2026-01-20 10:15:00 | 1136.90 | 2026-02-04 09:15:00 | 962.23 | TARGET | 174.67 |
