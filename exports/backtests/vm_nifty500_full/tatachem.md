# Tata Chemicals Ltd. (TATACHEM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 809.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 4 |
| ENTRY1 | 6 |
| ENTRY2 | 2 |
| EXIT | 6 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 5 / 3
- **Target hits / EMA400 exits:** 5 / 3
- **Total realized P&L (per unit):** 222.30
- **Avg P&L per closed trade:** 27.79

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-26 09:15:00 | 940.45 | 1026.73 | 1026.98 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-20 09:15:00 | 1043.10 | 997.53 | 997.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 10:15:00 | 1054.00 | 1001.84 | 999.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-18 09:15:00 | 1062.80 | 1073.02 | 1046.53 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-02-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 15:15:00 | 972.95 | 1032.66 | 1032.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-09 09:15:00 | 962.75 | 1031.97 | 1032.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-04 13:15:00 | 992.00 | 986.97 | 1003.53 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-03-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 14:15:00 | 1318.30 | 1019.40 | 1018.56 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 14:15:00 | 1002.45 | 1076.28 | 1076.51 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 12:15:00 | 1124.25 | 1076.33 | 1076.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 09:15:00 | 1138.35 | 1086.60 | 1081.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-26 15:15:00 | 1092.45 | 1094.53 | 1086.91 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-27 09:15:00 | 1095.20 | 1094.54 | 1086.96 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 1095.20 | 1094.54 | 1086.96 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-06-27 11:15:00 | 1085.65 | 1094.41 | 1086.97 | Close below EMA400 |

### Cycle 7 — SELL (started 2024-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 14:15:00 | 1060.55 | 1084.79 | 1084.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 1040.00 | 1084.10 | 1084.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-29 09:15:00 | 1090.40 | 1074.60 | 1079.08 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2024-08-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-02 11:15:00 | 1091.55 | 1082.97 | 1082.95 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 11:15:00 | 1061.15 | 1082.78 | 1082.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 15:15:00 | 1050.00 | 1081.73 | 1082.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 09:15:00 | 1066.60 | 1061.92 | 1070.47 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-06 14:15:00 | 1055.75 | 1072.73 | 1073.89 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 1059.85 | 1053.05 | 1061.81 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-09-26 09:15:00 | 1048.45 | 1053.35 | 1061.66 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-09-27 09:15:00 | 1072.45 | 1053.35 | 1061.37 | Close above EMA400 |

### Cycle 10 — BUY (started 2024-10-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 12:15:00 | 1144.05 | 1067.96 | 1067.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 09:15:00 | 1163.90 | 1075.73 | 1072.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 11:15:00 | 1088.60 | 1091.34 | 1081.46 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-21 09:15:00 | 1196.90 | 1090.58 | 1081.62 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-25 09:15:00 | 1086.75 | 1103.43 | 1089.83 | Close below EMA400 |

### Cycle 11 — SELL (started 2024-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 13:15:00 | 1038.20 | 1094.36 | 1094.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 14:15:00 | 1026.15 | 1093.68 | 1094.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 10:15:00 | 1092.30 | 1089.03 | 1091.71 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-24 14:15:00 | 1067.50 | 1088.49 | 1091.39 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-24 10:15:00 | 871.50 | 842.55 | 868.80 | Close above EMA400 |

### Cycle 12 — BUY (started 2025-06-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 15:15:00 | 905.60 | 865.43 | 865.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 10:15:00 | 906.50 | 866.20 | 865.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 904.10 | 905.89 | 890.45 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 10:15:00 | 917.75 | 906.25 | 891.24 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 906.65 | 920.39 | 905.84 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-14 09:15:00 | 919.75 | 920.25 | 905.91 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 937.45 | 947.92 | 933.06 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-14 15:15:00 | 932.50 | 947.63 | 933.06 | Close below EMA400 |

### Cycle 13 — SELL (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 11:15:00 | 907.65 | 940.70 | 940.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 12:15:00 | 907.05 | 940.37 | 940.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 13:15:00 | 775.85 | 774.87 | 808.63 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-08 10:15:00 | 768.40 | 774.99 | 808.03 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-10 15:15:00 | 705.00 | 656.90 | 687.25 | Close above EMA400 |

### Cycle 14 — BUY (started 2026-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 15:15:00 | 809.00 | 704.81 | 704.33 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-06-27 09:15:00 | 1095.20 | 2024-06-27 11:15:00 | 1085.65 | EXIT_EMA400 | -9.55 |
| SELL | 2024-09-06 14:15:00 | 1055.75 | 2024-09-19 11:15:00 | 1001.33 | TARGET | 54.42 |
| SELL | 2024-09-26 09:15:00 | 1048.45 | 2024-09-27 09:15:00 | 1072.45 | EXIT_EMA400 | -24.00 |
| BUY | 2024-10-21 09:15:00 | 1196.90 | 2024-10-25 09:15:00 | 1086.75 | EXIT_EMA400 | -110.15 |
| SELL | 2024-12-24 14:15:00 | 1067.50 | 2025-01-06 10:15:00 | 995.84 | TARGET | 71.66 |
| BUY | 2025-07-14 09:15:00 | 919.75 | 2025-07-22 14:15:00 | 961.27 | TARGET | 41.52 |
| BUY | 2025-06-24 10:15:00 | 917.75 | 2025-07-29 10:15:00 | 997.27 | TARGET | 79.52 |
| SELL | 2026-01-08 10:15:00 | 768.40 | 2026-03-18 15:15:00 | 649.52 | TARGET | 118.88 |
