# Aurobindo Pharma Ltd. (AUROPHARMA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1385.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 0 / 4
- **Target hits / EMA400 exits:** 0 / 4
- **Total realized P&L (per unit):** -185.80
- **Avg P&L per closed trade:** -46.45

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 10:15:00 | 1375.55 | 1453.74 | 1453.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 11:15:00 | 1365.90 | 1452.86 | 1453.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 09:15:00 | 1269.35 | 1268.62 | 1319.05 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-20 14:15:00 | 1240.15 | 1268.17 | 1317.57 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-31 09:15:00 | 1313.55 | 1267.80 | 1308.90 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 14:15:00 | 1208.00 | 1167.04 | 1166.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 09:15:00 | 1228.30 | 1168.09 | 1167.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 09:15:00 | 1181.50 | 1182.26 | 1175.38 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-13 09:15:00 | 1243.00 | 1181.49 | 1175.61 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 1184.80 | 1193.15 | 1183.64 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-22 12:15:00 | 1179.00 | 1193.01 | 1183.62 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 14:15:00 | 1137.80 | 1177.62 | 1177.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 09:15:00 | 1134.50 | 1168.35 | 1172.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 09:15:00 | 1143.40 | 1141.52 | 1155.67 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-30 12:15:00 | 1132.60 | 1141.33 | 1155.37 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 1148.40 | 1140.09 | 1153.98 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-07-02 11:15:00 | 1156.80 | 1140.36 | 1153.97 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 13:15:00 | 1103.20 | 1097.75 | 1097.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 09:15:00 | 1107.90 | 1097.90 | 1097.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 1097.50 | 1098.81 | 1098.29 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 10:15:00 | 1089.00 | 1097.74 | 1097.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 11:15:00 | 1084.60 | 1097.61 | 1097.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 14:15:00 | 1097.80 | 1097.40 | 1097.60 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 11:15:00 | 1113.70 | 1097.93 | 1097.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 09:15:00 | 1123.70 | 1099.09 | 1098.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 12:15:00 | 1187.00 | 1189.47 | 1160.31 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-12 13:15:00 | 1197.60 | 1186.00 | 1162.36 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-01 09:15:00 | 1173.40 | 1195.07 | 1176.25 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 1129.10 | 1173.45 | 1173.54 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 12:15:00 | 1191.20 | 1173.66 | 1173.61 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 1157.40 | 1173.41 | 1173.48 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 1228.60 | 1173.71 | 1173.63 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 1131.20 | 1174.73 | 1174.75 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 1222.50 | 1174.56 | 1174.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-09 13:15:00 | 1246.90 | 1190.02 | 1183.04 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-20 14:15:00 | 1240.15 | 2024-12-31 09:15:00 | 1313.55 | EXIT_EMA400 | -73.40 |
| BUY | 2025-05-13 09:15:00 | 1243.00 | 2025-05-22 12:15:00 | 1179.00 | EXIT_EMA400 | -64.00 |
| SELL | 2025-06-30 12:15:00 | 1132.60 | 2025-07-02 11:15:00 | 1156.80 | EXIT_EMA400 | -24.20 |
| BUY | 2025-12-12 13:15:00 | 1197.60 | 2026-01-01 09:15:00 | 1173.40 | EXIT_EMA400 | -24.20 |
