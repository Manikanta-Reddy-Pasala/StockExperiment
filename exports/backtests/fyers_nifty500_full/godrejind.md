# Godrej Industries Ltd. (GODREJIND.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 951.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** -118.04
- **Avg P&L per closed trade:** -19.67

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 14:15:00 | 981.25 | 1049.15 | 1049.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 09:15:00 | 970.50 | 1047.69 | 1048.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 10:15:00 | 1026.20 | 1025.74 | 1036.68 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-21 09:15:00 | 1001.05 | 1025.40 | 1036.19 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 14:15:00 | 1028.40 | 1023.47 | 1034.56 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-25 09:15:00 | 1042.25 | 1023.69 | 1034.56 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 09:15:00 | 1064.40 | 1042.53 | 1042.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 14:15:00 | 1082.65 | 1043.83 | 1043.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 10:15:00 | 1108.00 | 1111.07 | 1085.26 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-06 11:15:00 | 1140.75 | 1111.36 | 1085.54 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 15:15:00 | 1092.05 | 1111.10 | 1085.92 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-07 12:15:00 | 1084.50 | 1110.30 | 1086.01 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-16 09:15:00 | 964.30 | 1067.43 | 1067.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-16 10:15:00 | 958.35 | 1066.34 | 1067.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-20 10:15:00 | 911.05 | 893.26 | 946.38 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 15:15:00 | 1102.00 | 985.59 | 985.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-07 12:15:00 | 1113.90 | 998.18 | 991.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 09:15:00 | 1086.75 | 1101.30 | 1057.99 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-02 13:15:00 | 1118.00 | 1100.82 | 1060.05 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 15:15:00 | 1090.00 | 1102.06 | 1063.85 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-04-07 09:15:00 | 1037.00 | 1101.41 | 1063.72 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 15:15:00 | 1143.00 | 1176.65 | 1176.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 09:15:00 | 1139.20 | 1176.27 | 1176.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 13:15:00 | 1203.40 | 1137.86 | 1152.58 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 14:15:00 | 1285.00 | 1164.97 | 1164.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-22 09:15:00 | 1299.00 | 1175.51 | 1170.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 14:15:00 | 1198.40 | 1202.73 | 1186.97 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-05 15:15:00 | 1220.40 | 1201.29 | 1187.85 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 1198.50 | 1206.82 | 1193.86 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-09-17 13:15:00 | 1215.00 | 1206.70 | 1193.99 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-09-19 11:15:00 | 1190.00 | 1207.43 | 1195.11 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 1115.10 | 1188.82 | 1188.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 1107.60 | 1188.01 | 1188.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 12:15:00 | 1122.20 | 1118.41 | 1143.50 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-31 14:15:00 | 1104.60 | 1118.32 | 1143.20 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-07 14:15:00 | 1055.30 | 1020.46 | 1047.08 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-21 09:15:00 | 1001.05 | 2024-11-25 09:15:00 | 1042.25 | EXIT_EMA400 | -41.20 |
| BUY | 2025-01-06 11:15:00 | 1140.75 | 2025-01-07 12:15:00 | 1084.50 | EXIT_EMA400 | -56.25 |
| BUY | 2025-04-02 13:15:00 | 1118.00 | 2025-04-07 09:15:00 | 1037.00 | EXIT_EMA400 | -81.00 |
| BUY | 2025-09-05 15:15:00 | 1220.40 | 2025-09-19 11:15:00 | 1190.00 | EXIT_EMA400 | -30.40 |
| BUY | 2025-09-17 13:15:00 | 1215.00 | 2025-09-19 11:15:00 | 1190.00 | EXIT_EMA400 | -25.00 |
| SELL | 2025-10-31 14:15:00 | 1104.60 | 2025-12-19 10:15:00 | 988.79 | TARGET | 115.81 |
