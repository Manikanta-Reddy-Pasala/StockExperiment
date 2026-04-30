# Home First Finance Company India Ltd. (HOMEFIRST.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 1161.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT3 | 3 |
| ENTRY1 | 7 |
| ENTRY2 | 1 |
| EXIT | 7 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 3 / 5
- **Target hits / EMA400 exits:** 3 / 5
- **Total realized P&L (per unit):** 180.31
- **Avg P&L per closed trade:** 22.54

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 09:15:00 | 898.00 | 950.26 | 950.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 10:15:00 | 892.55 | 949.69 | 950.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-26 09:15:00 | 896.70 | 885.36 | 908.87 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-15 09:15:00 | 884.50 | 909.20 | 915.05 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-04-30 11:15:00 | 912.40 | 892.99 | 903.79 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-06-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-14 14:15:00 | 1071.00 | 878.27 | 878.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-14 15:15:00 | 1080.00 | 880.27 | 879.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 09:15:00 | 1028.85 | 1040.88 | 992.95 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-25 10:15:00 | 1046.00 | 1040.93 | 993.21 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 1027.70 | 1043.34 | 1004.94 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-08-05 11:15:00 | 1004.70 | 1042.62 | 1004.96 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-11-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-14 14:15:00 | 1052.00 | 1138.51 | 1138.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 13:15:00 | 1042.45 | 1133.09 | 1136.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 1151.70 | 1120.99 | 1129.27 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-27 09:15:00 | 1091.05 | 1120.37 | 1128.40 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-30 09:15:00 | 1085.75 | 1048.85 | 1077.50 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 1124.00 | 1007.90 | 1007.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 10:15:00 | 1138.50 | 1009.20 | 1008.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 10:15:00 | 1155.90 | 1157.10 | 1115.29 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-26 14:15:00 | 1175.70 | 1157.40 | 1119.03 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-29 09:15:00 | 1304.50 | 1363.61 | 1305.61 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 12:15:00 | 1255.20 | 1277.72 | 1277.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-02 13:15:00 | 1247.60 | 1277.42 | 1277.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 1273.80 | 1272.73 | 1275.09 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-09 11:15:00 | 1255.30 | 1272.25 | 1274.74 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-10 09:15:00 | 1275.00 | 1271.66 | 1274.38 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-09-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 15:15:00 | 1294.40 | 1276.71 | 1276.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 1307.60 | 1277.02 | 1276.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 10:15:00 | 1274.30 | 1278.20 | 1277.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-18 09:15:00 | 1300.70 | 1278.01 | 1277.40 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 1277.80 | 1278.26 | 1277.54 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-19 09:15:00 | 1269.00 | 1278.17 | 1277.50 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-09-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 13:15:00 | 1264.70 | 1276.75 | 1276.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 1264.60 | 1276.63 | 1276.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 11:15:00 | 1277.50 | 1275.55 | 1276.18 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-25 09:15:00 | 1266.00 | 1275.64 | 1276.21 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 1266.00 | 1275.64 | 1276.21 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-25 11:15:00 | 1248.20 | 1275.16 | 1275.97 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-10-15 10:15:00 | 1260.00 | 1241.79 | 1255.17 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-13 12:15:00 | 1194.20 | 1145.67 | 1145.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 12:15:00 | 1202.20 | 1156.52 | 1151.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 1151.10 | 1164.24 | 1156.12 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2026-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 12:15:00 | 1043.00 | 1148.19 | 1148.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-05 13:15:00 | 1038.50 | 1147.10 | 1148.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1060.60 | 1011.43 | 1059.21 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 13:15:00 | 1150.00 | 1085.03 | 1084.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 09:15:00 | 1157.35 | 1087.12 | 1085.96 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-04-15 09:15:00 | 884.50 | 2024-04-30 11:15:00 | 912.40 | EXIT_EMA400 | -27.90 |
| BUY | 2024-07-25 10:15:00 | 1046.00 | 2024-08-05 11:15:00 | 1004.70 | EXIT_EMA400 | -41.30 |
| SELL | 2024-11-27 09:15:00 | 1091.05 | 2024-12-02 09:15:00 | 979.00 | TARGET | 112.05 |
| BUY | 2025-05-26 14:15:00 | 1175.70 | 2025-06-24 09:15:00 | 1345.72 | TARGET | 170.02 |
| SELL | 2025-09-09 11:15:00 | 1255.30 | 2025-09-10 09:15:00 | 1275.00 | EXIT_EMA400 | -19.70 |
| BUY | 2025-09-18 09:15:00 | 1300.70 | 2025-09-19 09:15:00 | 1269.00 | EXIT_EMA400 | -31.70 |
| SELL | 2025-09-25 09:15:00 | 1266.00 | 2025-09-25 14:15:00 | 1235.36 | TARGET | 30.64 |
| SELL | 2025-09-25 11:15:00 | 1248.20 | 2025-10-15 10:15:00 | 1260.00 | EXIT_EMA400 | -11.80 |
