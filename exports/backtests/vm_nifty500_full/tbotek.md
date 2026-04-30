# TBO Tek Ltd. (TBOTEK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-15 09:15:00 → 2026-04-30 15:15:00 (3372 bars)
- **Last close:** 1255.40
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 5 |
| ENTRY1 | 5 |
| ENTRY2 | 3 |
| EXIT | 5 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 5 / 3
- **Target hits / EMA400 exits:** 4 / 4
- **Total realized P&L (per unit):** 567.65
- **Avg P&L per closed trade:** 70.96

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 12:15:00 | 1600.10 | 1718.14 | 1718.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 13:15:00 | 1590.15 | 1716.87 | 1717.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-07 09:15:00 | 1679.85 | 1663.62 | 1686.93 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-13 09:15:00 | 1612.90 | 1671.87 | 1688.35 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-11 11:15:00 | 1627.75 | 1578.78 | 1620.29 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 15:15:00 | 1748.65 | 1645.11 | 1645.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 11:15:00 | 1758.05 | 1653.65 | 1649.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-13 09:15:00 | 1639.85 | 1700.05 | 1677.05 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-14 10:15:00 | 1725.20 | 1697.44 | 1676.61 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 1725.20 | 1697.44 | 1676.61 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-17 13:15:00 | 1674.80 | 1696.81 | 1678.64 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 14:15:00 | 1623.20 | 1664.79 | 1664.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 10:15:00 | 1440.20 | 1582.20 | 1613.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-08 09:15:00 | 1172.40 | 1131.16 | 1236.00 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-05-09 09:15:00 | 1115.40 | 1132.86 | 1233.26 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 1216.70 | 1149.81 | 1226.24 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-05-16 10:15:00 | 1227.10 | 1150.58 | 1226.25 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-06-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 13:15:00 | 1365.00 | 1258.64 | 1258.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 14:15:00 | 1386.30 | 1277.59 | 1269.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 11:15:00 | 1317.90 | 1331.61 | 1303.53 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-10 13:15:00 | 1341.50 | 1331.60 | 1303.80 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1340.90 | 1371.09 | 1339.52 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-31 10:15:00 | 1358.00 | 1370.96 | 1339.61 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-08-05 09:15:00 | 1339.60 | 1374.54 | 1344.45 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 1478.30 | 1607.70 | 1607.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 11:15:00 | 1465.20 | 1606.28 | 1607.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 11:15:00 | 1516.70 | 1501.66 | 1542.10 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-12 09:15:00 | 1482.80 | 1509.95 | 1542.78 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 1482.80 | 1509.95 | 1542.78 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-13 09:15:00 | 1403.00 | 1506.80 | 1540.05 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 14:15:00 | 1538.30 | 1505.34 | 1538.49 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-16 09:15:00 | 1450.90 | 1504.95 | 1537.96 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-04-16 11:15:00 | 1267.80 | 1178.37 | 1264.95 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-13 09:15:00 | 1612.90 | 2024-12-11 11:15:00 | 1627.75 | EXIT_EMA400 | -14.85 |
| BUY | 2025-01-14 10:15:00 | 1725.20 | 2025-01-17 13:15:00 | 1674.80 | EXIT_EMA400 | -50.40 |
| SELL | 2025-05-09 09:15:00 | 1115.40 | 2025-05-16 10:15:00 | 1227.10 | EXIT_EMA400 | -111.70 |
| BUY | 2025-07-10 13:15:00 | 1341.50 | 2025-07-23 13:15:00 | 1454.60 | TARGET | 113.10 |
| BUY | 2025-07-31 10:15:00 | 1358.00 | 2025-08-01 10:15:00 | 1413.17 | TARGET | 55.17 |
| SELL | 2026-02-12 09:15:00 | 1482.80 | 2026-02-24 09:15:00 | 1302.86 | TARGET | 179.94 |
| SELL | 2026-02-16 09:15:00 | 1450.90 | 2026-03-02 09:15:00 | 1189.72 | TARGET | 261.18 |
| SELL | 2026-02-13 09:15:00 | 1403.00 | 2026-04-16 11:15:00 | 1267.80 | EXIT_EMA400 | 135.20 |
