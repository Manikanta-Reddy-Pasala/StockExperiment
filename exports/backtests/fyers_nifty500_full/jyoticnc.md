# Jyoti CNC Automation Ltd. (JYOTICNC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 751.95
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 5 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 6 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / EMA400 exits:** 2 / 5
- **Total realized P&L (per unit):** -77.21
- **Avg P&L per closed trade:** -11.03

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 11:15:00 | 1105.15 | 1147.27 | 1147.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 14:15:00 | 1086.20 | 1139.30 | 1143.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 09:15:00 | 1108.60 | 1086.13 | 1110.05 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-13 14:15:00 | 1045.05 | 1094.75 | 1110.70 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 1099.90 | 1093.58 | 1109.63 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-14 14:15:00 | 1120.00 | 1093.85 | 1109.68 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-11-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 15:15:00 | 1213.90 | 1122.12 | 1122.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 14:15:00 | 1250.00 | 1127.58 | 1124.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-09 14:15:00 | 1315.30 | 1320.26 | 1266.77 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 14:15:00 | 988.70 | 1237.01 | 1237.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 13:15:00 | 953.70 | 1131.79 | 1172.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-07 14:15:00 | 970.25 | 965.72 | 1054.10 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-11 09:15:00 | 929.70 | 966.93 | 1050.83 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 1042.00 | 969.49 | 1043.35 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-17 12:15:00 | 1022.00 | 970.69 | 1043.23 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-03-18 09:15:00 | 1048.45 | 972.87 | 1042.89 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 09:15:00 | 1145.00 | 1059.06 | 1058.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 15:15:00 | 1165.00 | 1064.41 | 1061.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 11:15:00 | 1206.50 | 1215.72 | 1163.56 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-09 09:15:00 | 1230.60 | 1215.72 | 1164.85 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 1174.90 | 1216.93 | 1170.14 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-12 11:15:00 | 1165.00 | 1215.19 | 1170.19 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 14:15:00 | 1052.50 | 1145.52 | 1145.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 09:15:00 | 1042.80 | 1143.58 | 1144.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 13:15:00 | 1076.60 | 1076.54 | 1103.56 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-22 10:15:00 | 1056.70 | 1076.21 | 1102.86 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-10 09:15:00 | 983.10 | 917.61 | 949.50 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 12:15:00 | 1008.55 | 942.12 | 941.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 14:15:00 | 1037.85 | 943.85 | 942.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 09:15:00 | 964.20 | 969.90 | 957.76 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-10 09:15:00 | 993.30 | 962.63 | 955.87 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 993.30 | 962.63 | 955.87 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-12-10 15:15:00 | 940.30 | 963.02 | 956.27 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 916.15 | 957.24 | 957.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 10:15:00 | 910.50 | 956.77 | 957.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-10 10:15:00 | 871.65 | 866.65 | 899.82 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-10 13:15:00 | 858.30 | 866.64 | 899.32 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 12:15:00 | 811.10 | 776.42 | 811.26 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-09 09:15:00 | 815.15 | 777.75 | 811.25 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-13 14:15:00 | 1045.05 | 2024-11-14 14:15:00 | 1120.00 | EXIT_EMA400 | -74.95 |
| SELL | 2025-03-11 09:15:00 | 929.70 | 2025-03-18 09:15:00 | 1048.45 | EXIT_EMA400 | -118.75 |
| SELL | 2025-03-17 12:15:00 | 1022.00 | 2025-03-18 09:15:00 | 1048.45 | EXIT_EMA400 | -26.45 |
| BUY | 2025-06-09 09:15:00 | 1230.60 | 2025-06-12 11:15:00 | 1165.00 | EXIT_EMA400 | -65.60 |
| SELL | 2025-07-22 10:15:00 | 1056.70 | 2025-08-11 10:15:00 | 918.23 | TARGET | 138.47 |
| BUY | 2025-12-10 09:15:00 | 993.30 | 2025-12-10 15:15:00 | 940.30 | EXIT_EMA400 | -53.00 |
| SELL | 2026-02-10 13:15:00 | 858.30 | 2026-03-13 09:15:00 | 735.23 | TARGET | 123.07 |
