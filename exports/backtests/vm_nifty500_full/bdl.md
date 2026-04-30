# Bharat Dynamics Ltd. (BDL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 1364.10
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / EMA400 exits:** 3 / 4
- **Total realized P&L (per unit):** 150.29
- **Avg P&L per closed trade:** 21.47

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-09-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-15 13:15:00 | 536.75 | 569.88 | 569.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-15 14:15:00 | 532.90 | 569.51 | 569.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-03 09:15:00 | 504.38 | 504.00 | 522.44 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-11-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 13:15:00 | 575.00 | 531.21 | 531.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-24 14:15:00 | 580.00 | 531.69 | 531.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-12 11:15:00 | 836.00 | 849.27 | 783.92 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-02-16 13:15:00 | 870.80 | 840.79 | 788.36 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-03-13 09:15:00 | 822.92 | 879.37 | 835.99 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-08-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 15:15:00 | 1311.40 | 1387.73 | 1388.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 09:15:00 | 1302.60 | 1382.25 | 1385.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-10 10:15:00 | 1202.95 | 1193.93 | 1255.01 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-15 09:15:00 | 1172.55 | 1196.48 | 1250.57 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-27 14:15:00 | 1116.95 | 1050.50 | 1116.34 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 13:15:00 | 1280.00 | 1152.83 | 1152.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 09:15:00 | 1340.35 | 1173.79 | 1167.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-27 09:15:00 | 1184.20 | 1201.30 | 1183.72 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-28 12:15:00 | 1241.95 | 1200.41 | 1184.12 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-02-03 09:15:00 | 1170.00 | 1214.56 | 1193.45 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 09:15:00 | 1037.55 | 1182.11 | 1182.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-18 10:15:00 | 1029.80 | 1180.60 | 1181.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-07 09:15:00 | 1119.05 | 1092.16 | 1127.70 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-25 10:15:00 | 1307.05 | 1146.57 | 1146.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-25 13:15:00 | 1317.65 | 1151.48 | 1148.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 14:15:00 | 1853.00 | 1853.16 | 1710.45 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-27 12:15:00 | 1866.90 | 1849.51 | 1721.45 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-17 09:15:00 | 1790.00 | 1882.78 | 1793.19 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 15:15:00 | 1590.50 | 1741.64 | 1741.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 09:15:00 | 1574.50 | 1739.98 | 1740.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 11:15:00 | 1570.00 | 1527.41 | 1593.54 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-26 12:15:00 | 1521.60 | 1560.57 | 1593.69 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 1539.00 | 1532.71 | 1563.94 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-20 09:15:00 | 1537.30 | 1533.17 | 1563.41 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 1550.20 | 1533.69 | 1561.36 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-24 10:15:00 | 1539.40 | 1533.75 | 1561.25 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 1539.90 | 1516.66 | 1543.17 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-11-12 09:15:00 | 1552.50 | 1517.94 | 1542.91 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-02-16 13:15:00 | 870.80 | 2024-03-13 09:15:00 | 822.92 | EXIT_EMA400 | -47.88 |
| SELL | 2024-10-15 09:15:00 | 1172.55 | 2024-11-18 15:15:00 | 938.50 | TARGET | 234.05 |
| BUY | 2025-01-28 12:15:00 | 1241.95 | 2025-02-03 09:15:00 | 1170.00 | EXIT_EMA400 | -71.95 |
| BUY | 2025-06-27 12:15:00 | 1866.90 | 2025-07-17 09:15:00 | 1790.00 | EXIT_EMA400 | -76.90 |
| SELL | 2025-10-20 09:15:00 | 1537.30 | 2025-11-06 09:15:00 | 1458.98 | TARGET | 78.32 |
| SELL | 2025-10-24 10:15:00 | 1539.40 | 2025-11-06 09:15:00 | 1473.86 | TARGET | 65.54 |
| SELL | 2025-09-26 12:15:00 | 1521.60 | 2025-11-12 09:15:00 | 1552.50 | EXIT_EMA400 | -30.90 |
