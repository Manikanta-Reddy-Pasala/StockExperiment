# Adani Ports and Special Economic Zone Ltd. (ADANIPORTS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 1657.30
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT3 | 2 |
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| EXIT | 6 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 2 / 4
- **Total realized P&L (per unit):** 113.39
- **Avg P&L per closed trade:** 18.90

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-02 12:15:00 | 777.85 | 798.85 | 798.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-02 14:15:00 | 772.40 | 798.38 | 798.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-06 09:15:00 | 803.45 | 797.88 | 798.39 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-07 13:15:00 | 796.95 | 798.31 | 798.58 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 13:15:00 | 796.95 | 798.31 | 798.58 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2023-11-07 15:15:00 | 799.15 | 798.30 | 798.57 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-11-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 12:15:00 | 821.95 | 798.96 | 798.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-28 10:15:00 | 833.50 | 802.67 | 801.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-13 09:15:00 | 1262.90 | 1282.85 | 1206.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-03-26 13:15:00 | 1310.45 | 1273.19 | 1220.44 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-05-06 09:15:00 | 1265.15 | 1321.52 | 1286.67 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 15:15:00 | 1410.90 | 1463.18 | 1463.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 09:15:00 | 1385.90 | 1454.86 | 1458.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 10:15:00 | 1411.00 | 1399.66 | 1422.94 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-31 10:15:00 | 1383.70 | 1399.55 | 1422.08 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-06 12:15:00 | 1142.10 | 1102.39 | 1141.14 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 11:15:00 | 1215.20 | 1155.55 | 1155.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 12:15:00 | 1219.80 | 1156.19 | 1155.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 1396.70 | 1397.69 | 1333.28 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-26 09:15:00 | 1408.30 | 1388.04 | 1343.50 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-25 15:15:00 | 1391.90 | 1423.93 | 1392.62 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 15:15:00 | 1301.40 | 1376.60 | 1376.85 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 14:15:00 | 1406.90 | 1368.35 | 1368.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 09:15:00 | 1409.40 | 1369.13 | 1368.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-29 10:15:00 | 1387.90 | 1391.38 | 1381.44 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-30 14:15:00 | 1404.00 | 1391.54 | 1382.05 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-12-29 09:15:00 | 1466.90 | 1493.53 | 1473.75 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-01-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 15:15:00 | 1403.00 | 1465.15 | 1465.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 10:15:00 | 1395.90 | 1463.86 | 1464.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 1502.30 | 1428.77 | 1443.91 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 1562.70 | 1457.76 | 1457.39 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 1362.40 | 1476.40 | 1476.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 1352.40 | 1448.69 | 1461.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1457.10 | 1403.09 | 1430.63 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-08 11:15:00 | 1452.50 | 1404.14 | 1430.88 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 11:15:00 | 1452.50 | 1404.14 | 1430.88 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-08 12:15:00 | 1456.20 | 1404.65 | 1431.00 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 12:15:00 | 1600.80 | 1450.09 | 1449.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 09:15:00 | 1629.40 | 1481.25 | 1466.34 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-11-07 13:15:00 | 796.95 | 2023-11-07 15:15:00 | 799.15 | EXIT_EMA400 | -2.20 |
| BUY | 2024-03-26 13:15:00 | 1310.45 | 2024-05-06 09:15:00 | 1265.15 | EXIT_EMA400 | -45.30 |
| SELL | 2024-10-31 10:15:00 | 1383.70 | 2024-11-14 11:15:00 | 1268.55 | TARGET | 115.15 |
| BUY | 2025-06-26 09:15:00 | 1408.30 | 2025-07-25 15:15:00 | 1391.90 | EXIT_EMA400 | -16.40 |
| BUY | 2025-09-30 14:15:00 | 1404.00 | 2025-10-16 09:15:00 | 1469.84 | TARGET | 65.84 |
| SELL | 2026-04-08 11:15:00 | 1452.50 | 2026-04-08 12:15:00 | 1456.20 | EXIT_EMA400 | -3.70 |
