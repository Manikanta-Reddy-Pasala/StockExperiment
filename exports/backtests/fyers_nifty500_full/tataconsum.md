# Tata Consumer Products Ltd. (TATACONSUM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1147.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 3 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 6 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 1 / 6
- **Target hits / EMA400 exits:** 1 / 6
- **Total realized P&L (per unit):** -92.91
- **Avg P&L per closed trade:** -13.27

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 14:15:00 | 1112.80 | 1171.89 | 1172.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 10:15:00 | 1111.80 | 1170.15 | 1171.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 10:15:00 | 934.30 | 932.79 | 975.07 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-16 10:15:00 | 929.70 | 944.96 | 970.61 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-21 09:15:00 | 969.70 | 946.11 | 968.75 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 10:15:00 | 1018.00 | 980.20 | 980.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-06 12:15:00 | 1020.65 | 980.96 | 980.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 1000.50 | 1002.45 | 993.49 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-02-21 10:15:00 | 1009.60 | 1002.52 | 993.57 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 13:15:00 | 997.00 | 1002.46 | 993.67 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-02-21 14:15:00 | 1005.45 | 1002.49 | 993.73 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 1005.45 | 1002.53 | 993.84 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-02-28 10:15:00 | 992.20 | 1002.88 | 994.94 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-03-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 14:15:00 | 962.05 | 988.42 | 988.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 957.75 | 986.70 | 987.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-24 11:15:00 | 973.60 | 971.81 | 978.65 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-25 10:15:00 | 963.75 | 971.76 | 978.42 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-27 15:15:00 | 977.45 | 970.95 | 977.39 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 11:15:00 | 1069.40 | 983.33 | 983.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 12:15:00 | 1071.00 | 984.20 | 983.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 1093.70 | 1101.70 | 1061.27 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-09 11:15:00 | 1109.30 | 1101.71 | 1061.68 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-12 10:15:00 | 1091.20 | 1118.66 | 1097.45 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 15:15:00 | 1064.00 | 1093.61 | 1093.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 1061.20 | 1092.08 | 1092.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 1081.30 | 1070.87 | 1079.53 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-29 09:15:00 | 1056.60 | 1075.64 | 1080.21 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-02 09:15:00 | 1090.10 | 1074.79 | 1079.45 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 14:15:00 | 1102.90 | 1082.45 | 1082.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 09:15:00 | 1120.20 | 1083.89 | 1083.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 10:15:00 | 1114.90 | 1114.92 | 1104.03 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-15 13:15:00 | 1119.00 | 1115.10 | 1104.65 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 1149.30 | 1162.92 | 1147.12 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-12-03 10:15:00 | 1144.20 | 1162.73 | 1147.10 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 14:15:00 | 1124.50 | 1161.68 | 1161.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 1115.20 | 1156.69 | 1158.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-09 11:15:00 | 1082.30 | 1076.96 | 1104.10 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 11:15:00 | 1148.60 | 1117.10 | 1116.98 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-16 10:15:00 | 929.70 | 2025-01-21 09:15:00 | 969.70 | EXIT_EMA400 | -40.00 |
| BUY | 2025-02-21 10:15:00 | 1009.60 | 2025-02-28 10:15:00 | 992.20 | EXIT_EMA400 | -17.40 |
| BUY | 2025-02-21 14:15:00 | 1005.45 | 2025-02-28 10:15:00 | 992.20 | EXIT_EMA400 | -13.25 |
| SELL | 2025-03-25 10:15:00 | 963.75 | 2025-03-27 15:15:00 | 977.45 | EXIT_EMA400 | -13.70 |
| BUY | 2025-05-09 11:15:00 | 1109.30 | 2025-06-12 10:15:00 | 1091.20 | EXIT_EMA400 | -18.10 |
| SELL | 2025-08-29 09:15:00 | 1056.60 | 2025-09-02 09:15:00 | 1090.10 | EXIT_EMA400 | -33.50 |
| BUY | 2025-10-15 13:15:00 | 1119.00 | 2025-10-17 10:15:00 | 1162.04 | TARGET | 43.04 |
