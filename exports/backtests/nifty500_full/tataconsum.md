# Tata Consumer Products Ltd. (TATACONSUM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 1144.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT3 | 5 |
| ENTRY1 | 9 |
| ENTRY2 | 0 |
| EXIT | 9 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 3 / 6
- **Target hits / EMA400 exits:** 3 / 6
- **Total realized P&L (per unit):** 22.22
- **Avg P&L per closed trade:** 2.47

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 13:15:00 | 1100.00 | 1123.60 | 1123.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 10:15:00 | 1091.95 | 1121.97 | 1122.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 11:15:00 | 1113.45 | 1107.55 | 1114.29 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-24 09:15:00 | 1103.05 | 1108.37 | 1114.32 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-06-05 09:15:00 | 1151.00 | 1095.58 | 1105.72 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 11:15:00 | 1146.50 | 1111.08 | 1110.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 10:15:00 | 1170.50 | 1122.91 | 1117.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 09:15:00 | 1166.95 | 1176.42 | 1155.36 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-16 12:15:00 | 1187.85 | 1175.97 | 1157.50 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 1176.45 | 1189.53 | 1173.17 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-09-06 14:15:00 | 1173.05 | 1188.87 | 1173.24 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 12:15:00 | 1122.00 | 1177.80 | 1178.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 14:15:00 | 1113.00 | 1176.57 | 1177.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 10:15:00 | 934.30 | 932.90 | 975.78 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-16 10:15:00 | 929.70 | 945.08 | 971.17 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-21 09:15:00 | 969.70 | 946.21 | 969.25 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-10 09:15:00 | 1031.50 | 980.77 | 980.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-10 10:15:00 | 1036.25 | 981.32 | 980.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 12:15:00 | 995.95 | 1000.29 | 992.18 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-02-24 12:15:00 | 1009.85 | 1000.60 | 992.61 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 09:15:00 | 995.00 | 1001.27 | 993.65 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-02-28 10:15:00 | 992.20 | 1001.18 | 993.65 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-03-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 14:15:00 | 962.05 | 987.27 | 987.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 957.75 | 985.61 | 986.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-24 11:15:00 | 973.70 | 971.20 | 977.88 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-25 10:15:00 | 963.75 | 971.18 | 977.67 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-27 15:15:00 | 980.40 | 970.49 | 976.72 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-04-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 10:15:00 | 1066.90 | 982.09 | 982.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 12:15:00 | 1071.00 | 983.83 | 982.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 1093.70 | 1101.67 | 1061.03 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-09 11:15:00 | 1109.30 | 1101.68 | 1061.44 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-12 10:15:00 | 1091.60 | 1118.63 | 1097.34 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 09:15:00 | 1070.40 | 1093.37 | 1093.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 1061.20 | 1092.08 | 1092.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 1081.30 | 1070.88 | 1079.52 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-18 11:15:00 | 1073.00 | 1070.99 | 1079.48 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 11:15:00 | 1073.00 | 1070.99 | 1079.48 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-08-19 12:15:00 | 1083.30 | 1071.30 | 1079.31 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 14:15:00 | 1102.70 | 1082.44 | 1082.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 09:15:00 | 1120.20 | 1083.85 | 1083.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 12:15:00 | 1114.80 | 1114.88 | 1103.73 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-16 11:15:00 | 1122.40 | 1115.19 | 1104.95 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 1149.20 | 1163.06 | 1147.22 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-12-03 10:15:00 | 1144.20 | 1162.87 | 1147.20 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-09 10:15:00 | 1160.40 | 1161.93 | 1161.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-09 11:15:00 | 1157.40 | 1161.89 | 1161.91 | Break + close below crossover candle low |

### Cycle 10 — BUY (started 2026-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 12:15:00 | 1169.90 | 1161.97 | 1161.95 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2026-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 12:15:00 | 1153.60 | 1161.91 | 1161.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-10 13:15:00 | 1152.00 | 1161.81 | 1161.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 10:15:00 | 1157.90 | 1156.44 | 1158.94 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-27 09:15:00 | 1142.70 | 1160.07 | 1160.43 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 1102.40 | 1078.85 | 1102.88 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-15 10:15:00 | 1104.90 | 1079.11 | 1102.89 | Close above EMA400 |

### Cycle 12 — BUY (started 2026-04-30 15:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 15:30:00 | 1144.60 | 1117.75 | 1117.65 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-05-24 09:15:00 | 1103.05 | 2024-05-29 15:15:00 | 1069.25 | TARGET | 33.80 |
| BUY | 2024-08-16 12:15:00 | 1187.85 | 2024-09-06 14:15:00 | 1173.05 | EXIT_EMA400 | -14.80 |
| SELL | 2025-01-16 10:15:00 | 929.70 | 2025-01-21 09:15:00 | 969.70 | EXIT_EMA400 | -40.00 |
| BUY | 2025-02-24 12:15:00 | 1009.85 | 2025-02-28 10:15:00 | 992.20 | EXIT_EMA400 | -17.65 |
| SELL | 2025-03-25 10:15:00 | 963.75 | 2025-03-27 15:15:00 | 980.40 | EXIT_EMA400 | -16.65 |
| BUY | 2025-05-09 11:15:00 | 1109.30 | 2025-06-12 10:15:00 | 1091.60 | EXIT_EMA400 | -17.70 |
| SELL | 2025-08-18 11:15:00 | 1073.00 | 2025-08-19 12:15:00 | 1083.30 | EXIT_EMA400 | -10.30 |
| BUY | 2025-10-16 11:15:00 | 1122.40 | 2025-10-20 09:15:00 | 1174.74 | TARGET | 52.34 |
| SELL | 2026-02-27 09:15:00 | 1142.70 | 2026-03-09 09:15:00 | 1089.52 | TARGET | 53.18 |
