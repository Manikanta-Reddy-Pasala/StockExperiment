# Godrej Industries Ltd. (GODREJIND.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 959.05
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 4 |
| ENTRY1 | 6 |
| ENTRY2 | 2 |
| EXIT | 6 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 2 / 6
- **Target hits / EMA400 exits:** 2 / 6
- **Total realized P&L (per unit):** -35.78
- **Avg P&L per closed trade:** -4.47

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 12:15:00 | 797.60 | 817.63 | 817.66 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-06-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 12:15:00 | 830.90 | 817.69 | 817.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-13 14:15:00 | 833.50 | 817.97 | 817.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-18 09:15:00 | 814.25 | 819.00 | 818.36 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-24 10:15:00 | 846.35 | 818.49 | 818.17 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 12:15:00 | 870.65 | 889.47 | 868.14 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-08-05 13:15:00 | 879.75 | 889.37 | 868.20 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-08-06 15:15:00 | 868.00 | 888.69 | 868.78 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 14:15:00 | 981.25 | 1049.61 | 1049.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 09:15:00 | 970.50 | 1048.14 | 1049.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 10:15:00 | 1026.20 | 1026.18 | 1037.01 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-21 09:15:00 | 1001.05 | 1025.89 | 1036.55 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 14:15:00 | 1028.40 | 1023.95 | 1034.92 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-25 09:15:00 | 1042.25 | 1024.17 | 1034.92 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 10:15:00 | 1066.20 | 1042.91 | 1042.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 14:15:00 | 1082.40 | 1043.96 | 1043.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 10:15:00 | 1108.00 | 1111.19 | 1085.40 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-06 11:15:00 | 1140.75 | 1111.49 | 1085.68 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 1086.70 | 1110.70 | 1086.16 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-07 12:15:00 | 1084.50 | 1110.43 | 1086.16 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-16 09:15:00 | 964.30 | 1067.61 | 1067.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-16 10:15:00 | 958.35 | 1066.52 | 1067.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-20 10:15:00 | 911.05 | 894.97 | 948.71 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 09:15:00 | 1108.35 | 988.04 | 987.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-07 12:15:00 | 1113.90 | 999.29 | 993.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 09:15:00 | 1086.75 | 1101.83 | 1059.18 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-02 13:15:00 | 1118.00 | 1101.34 | 1061.19 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-04-07 09:15:00 | 1037.00 | 1101.90 | 1064.80 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 15:15:00 | 1144.40 | 1176.89 | 1176.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 09:15:00 | 1139.20 | 1176.51 | 1176.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 13:15:00 | 1204.70 | 1137.92 | 1152.71 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 14:15:00 | 1285.00 | 1165.11 | 1165.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-22 09:15:00 | 1299.00 | 1175.61 | 1170.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 14:15:00 | 1198.40 | 1202.75 | 1187.06 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-09 09:15:00 | 1210.30 | 1201.41 | 1188.50 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 1198.50 | 1206.70 | 1193.84 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-09-17 13:15:00 | 1215.50 | 1206.59 | 1193.97 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-09-19 11:15:00 | 1190.00 | 1207.33 | 1195.10 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 1115.10 | 1188.65 | 1188.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 1107.60 | 1187.85 | 1188.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 12:15:00 | 1122.20 | 1118.36 | 1143.46 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-31 14:15:00 | 1104.60 | 1118.28 | 1143.16 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-07 14:15:00 | 1055.30 | 1020.33 | 1047.00 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-06-24 10:15:00 | 846.35 | 2024-08-01 10:15:00 | 930.88 | TARGET | 84.53 |
| BUY | 2024-08-05 13:15:00 | 879.75 | 2024-08-06 15:15:00 | 868.00 | EXIT_EMA400 | -11.75 |
| SELL | 2024-11-21 09:15:00 | 1001.05 | 2024-11-25 09:15:00 | 1042.25 | EXIT_EMA400 | -41.20 |
| BUY | 2025-01-06 11:15:00 | 1140.75 | 2025-01-07 12:15:00 | 1084.50 | EXIT_EMA400 | -56.25 |
| BUY | 2025-04-02 13:15:00 | 1118.00 | 2025-04-07 09:15:00 | 1037.00 | EXIT_EMA400 | -81.00 |
| BUY | 2025-09-09 09:15:00 | 1210.30 | 2025-09-19 11:15:00 | 1190.00 | EXIT_EMA400 | -20.30 |
| BUY | 2025-09-17 13:15:00 | 1215.50 | 2025-09-19 11:15:00 | 1190.00 | EXIT_EMA400 | -25.50 |
| SELL | 2025-10-31 14:15:00 | 1104.60 | 2025-12-19 10:15:00 | 988.91 | TARGET | 115.69 |
