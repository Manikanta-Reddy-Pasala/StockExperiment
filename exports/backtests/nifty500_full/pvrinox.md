# PVR INOX Ltd. (PVRINOX.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 1068.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / EMA400 exits:** 3 / 4
- **Total realized P&L (per unit):** 344.59
- **Avg P&L per closed trade:** 49.23

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-09 12:15:00 | 1579.00 | 1689.14 | 1689.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-09 13:15:00 | 1574.70 | 1688.00 | 1688.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 15:15:00 | 1419.95 | 1417.11 | 1480.01 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-03-11 09:15:00 | 1394.35 | 1416.89 | 1479.58 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-04-10 13:15:00 | 1417.45 | 1369.81 | 1412.89 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 10:15:00 | 1456.40 | 1372.26 | 1372.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 11:15:00 | 1469.95 | 1378.65 | 1375.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 09:15:00 | 1414.00 | 1430.85 | 1409.04 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-22 11:15:00 | 1432.65 | 1428.32 | 1409.40 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-07-23 09:15:00 | 1392.95 | 1428.00 | 1409.71 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-11-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 12:15:00 | 1470.05 | 1554.52 | 1554.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 14:15:00 | 1463.00 | 1547.37 | 1550.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 13:15:00 | 1513.70 | 1507.96 | 1526.64 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-09 14:15:00 | 1476.00 | 1527.98 | 1533.48 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 1523.35 | 1511.09 | 1523.36 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-18 09:15:00 | 1451.65 | 1509.61 | 1522.19 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-04-24 12:15:00 | 1015.85 | 951.59 | 1005.91 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 15:15:00 | 1015.00 | 985.13 | 985.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 12:15:00 | 1024.45 | 986.36 | 985.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 11:15:00 | 987.35 | 989.62 | 987.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-30 09:15:00 | 998.75 | 988.77 | 987.16 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 987.70 | 988.92 | 987.28 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-30 15:15:00 | 986.50 | 988.90 | 987.27 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 13:15:00 | 1088.40 | 1106.57 | 1106.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 15:15:00 | 1080.50 | 1106.14 | 1106.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 10:15:00 | 1119.90 | 1095.93 | 1100.93 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-16 09:15:00 | 1076.00 | 1095.58 | 1100.61 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 14:15:00 | 1015.00 | 996.14 | 1027.05 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-05 13:15:00 | 988.10 | 996.96 | 1026.55 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-02-09 10:15:00 | 1026.35 | 997.06 | 1025.01 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-03-11 09:15:00 | 1394.35 | 2024-04-10 13:15:00 | 1417.45 | EXIT_EMA400 | -23.10 |
| BUY | 2024-07-22 11:15:00 | 1432.65 | 2024-07-23 09:15:00 | 1392.95 | EXIT_EMA400 | -39.70 |
| SELL | 2024-12-09 14:15:00 | 1476.00 | 2024-12-31 09:15:00 | 1303.56 | TARGET | 172.44 |
| SELL | 2024-12-18 09:15:00 | 1451.65 | 2025-01-07 09:15:00 | 1240.02 | TARGET | 211.63 |
| BUY | 2025-07-30 09:15:00 | 998.75 | 2025-07-30 15:15:00 | 986.50 | EXIT_EMA400 | -12.25 |
| SELL | 2025-12-16 09:15:00 | 1076.00 | 2025-12-26 11:15:00 | 1002.18 | TARGET | 73.82 |
| SELL | 2026-02-05 13:15:00 | 988.10 | 2026-02-09 10:15:00 | 1026.35 | EXIT_EMA400 | -38.25 |
