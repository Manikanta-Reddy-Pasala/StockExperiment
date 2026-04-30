# Carborundum Universal Ltd. (CARBORUNIV.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 951.75
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 15 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 4 |
| EXIT | 5 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 3 / 6
- **Target hits / EMA400 exits:** 2 / 7
- **Total realized P&L (per unit):** -57.53
- **Avg P&L per closed trade:** -6.39

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-18 15:15:00 | 1181.45 | 1169.68 | 1169.65 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-09-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 11:15:00 | 1150.30 | 1169.44 | 1169.53 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-22 10:15:00 | 1243.35 | 1169.76 | 1169.67 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2023-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-11 09:15:00 | 1155.05 | 1170.82 | 1170.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 14:15:00 | 1144.00 | 1169.67 | 1170.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-17 09:15:00 | 1127.10 | 1106.98 | 1129.32 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-20 12:15:00 | 1097.75 | 1107.13 | 1128.32 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2023-11-22 12:15:00 | 1163.30 | 1106.29 | 1126.44 | Close above EMA400 |

### Cycle 5 — BUY (started 2023-12-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 15:15:00 | 1207.95 | 1139.63 | 1139.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-12 09:15:00 | 1216.00 | 1147.68 | 1143.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 10:15:00 | 1157.00 | 1163.67 | 1153.29 | EMA200 retest candle locked |

### Cycle 6 — SELL (started 2023-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-29 11:15:00 | 1111.80 | 1144.88 | 1144.93 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-01-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-15 12:15:00 | 1183.45 | 1143.40 | 1143.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-16 09:15:00 | 1192.50 | 1145.18 | 1144.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 11:15:00 | 1141.10 | 1146.30 | 1144.86 | EMA200 retest candle locked |

### Cycle 8 — SELL (started 2024-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 14:15:00 | 1125.80 | 1143.56 | 1143.59 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-01-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 09:15:00 | 1195.00 | 1143.89 | 1143.76 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 11:15:00 | 1131.20 | 1143.84 | 1143.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-01 15:15:00 | 1127.75 | 1143.33 | 1143.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-07 09:15:00 | 1170.85 | 1137.59 | 1140.58 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-02-09 09:15:00 | 1127.15 | 1138.29 | 1140.74 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 1127.15 | 1138.29 | 1140.74 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-02-09 10:15:00 | 1119.95 | 1138.10 | 1140.64 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 13:15:00 | 1139.00 | 1137.85 | 1140.47 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-02-09 14:15:00 | 1128.30 | 1137.75 | 1140.41 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-02-12 09:15:00 | 1144.00 | 1137.73 | 1140.37 | Close above EMA400 |

### Cycle 11 — BUY (started 2024-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-27 11:15:00 | 1249.15 | 1119.72 | 1119.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-28 09:15:00 | 1275.00 | 1126.12 | 1122.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 14:15:00 | 1516.95 | 1535.34 | 1425.34 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-06 09:15:00 | 1583.00 | 1536.09 | 1430.56 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 1665.65 | 1687.41 | 1613.61 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-07-24 09:15:00 | 1701.00 | 1687.22 | 1614.98 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-08-05 09:15:00 | 1631.90 | 1699.91 | 1640.11 | Close below EMA400 |

### Cycle 12 — SELL (started 2024-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 10:15:00 | 1529.90 | 1607.40 | 1607.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 11:15:00 | 1526.05 | 1606.59 | 1607.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 09:15:00 | 1522.50 | 1493.41 | 1525.91 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-17 11:15:00 | 1464.50 | 1494.13 | 1523.81 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-18 09:15:00 | 1499.50 | 1436.39 | 1470.01 | Close above EMA400 |

### Cycle 13 — BUY (started 2025-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 12:15:00 | 1002.10 | 949.82 | 949.67 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 10:15:00 | 925.00 | 950.60 | 950.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 11:15:00 | 916.25 | 950.26 | 950.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 14:15:00 | 938.25 | 938.16 | 943.64 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-13 09:15:00 | 923.15 | 938.00 | 943.51 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 932.85 | 923.07 | 932.98 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-31 09:15:00 | 912.95 | 923.06 | 932.53 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-11-26 11:15:00 | 909.10 | 883.80 | 904.82 | Close above EMA400 |

### Cycle 15 — BUY (started 2026-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 12:15:00 | 891.00 | 819.08 | 818.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 907.70 | 826.36 | 822.70 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-11-20 12:15:00 | 1097.75 | 2023-11-22 12:15:00 | 1163.30 | EXIT_EMA400 | -65.55 |
| SELL | 2024-02-09 09:15:00 | 1127.15 | 2024-02-12 09:15:00 | 1144.00 | EXIT_EMA400 | -16.85 |
| SELL | 2024-02-09 10:15:00 | 1119.95 | 2024-02-12 09:15:00 | 1144.00 | EXIT_EMA400 | -24.05 |
| SELL | 2024-02-09 14:15:00 | 1128.30 | 2024-02-12 09:15:00 | 1144.00 | EXIT_EMA400 | -15.70 |
| BUY | 2024-06-06 09:15:00 | 1583.00 | 2024-08-05 09:15:00 | 1631.90 | EXIT_EMA400 | 48.90 |
| BUY | 2024-07-24 09:15:00 | 1701.00 | 2024-08-05 09:15:00 | 1631.90 | EXIT_EMA400 | -69.10 |
| SELL | 2024-10-17 11:15:00 | 1464.50 | 2024-11-18 09:15:00 | 1499.50 | EXIT_EMA400 | -35.00 |
| SELL | 2025-10-13 09:15:00 | 923.15 | 2025-11-18 09:15:00 | 862.08 | TARGET | 61.07 |
| SELL | 2025-10-31 09:15:00 | 912.95 | 2025-11-18 14:15:00 | 854.20 | TARGET | 58.75 |
