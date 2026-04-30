# Axis Bank Ltd. (AXISBANK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1275.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 6 |
| ENTRY1 | 4 |
| ENTRY2 | 5 |
| EXIT | 4 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 4 / 5
- **Target hits / EMA400 exits:** 3 / 6
- **Total realized P&L (per unit):** -2.94
- **Avg P&L per closed trade:** -0.33

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 11:15:00 | 1149.85 | 1200.84 | 1200.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-09 12:15:00 | 1145.50 | 1200.29 | 1200.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 12:15:00 | 1182.20 | 1181.87 | 1189.32 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-28 09:15:00 | 1174.50 | 1181.77 | 1189.12 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-09-02 13:15:00 | 1191.40 | 1180.70 | 1187.67 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 11:15:00 | 1242.50 | 1190.36 | 1190.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-18 12:15:00 | 1245.85 | 1190.91 | 1190.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 09:15:00 | 1201.15 | 1219.10 | 1207.13 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 11:15:00 | 1162.20 | 1198.30 | 1198.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-15 12:15:00 | 1156.65 | 1197.89 | 1198.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 11:15:00 | 1194.45 | 1189.53 | 1193.69 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-22 12:15:00 | 1181.00 | 1189.95 | 1193.62 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 1181.60 | 1186.59 | 1191.55 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-10-25 11:15:00 | 1171.60 | 1186.36 | 1191.38 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 15:15:00 | 1189.00 | 1186.21 | 1191.21 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-10-28 09:15:00 | 1183.60 | 1186.18 | 1191.17 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 14:15:00 | 1186.10 | 1185.16 | 1190.35 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-10-29 15:15:00 | 1183.35 | 1185.14 | 1190.32 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 1181.10 | 1175.23 | 1183.29 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-12 12:15:00 | 1161.40 | 1175.05 | 1183.08 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 1164.00 | 1155.09 | 1167.63 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-04 13:15:00 | 1156.10 | 1155.22 | 1167.45 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 11:15:00 | 1166.90 | 1155.43 | 1167.25 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-05 12:15:00 | 1168.90 | 1155.56 | 1167.26 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 15:15:00 | 1098.00 | 1044.57 | 1044.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 10:15:00 | 1106.90 | 1060.18 | 1053.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 12:15:00 | 1177.10 | 1180.86 | 1149.00 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-06 10:15:00 | 1192.40 | 1179.09 | 1150.97 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-01 09:15:00 | 1174.60 | 1205.86 | 1180.78 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 13:15:00 | 1101.00 | 1169.99 | 1170.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 14:15:00 | 1097.50 | 1169.27 | 1169.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 10:15:00 | 1074.20 | 1073.36 | 1097.24 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 11:15:00 | 1129.80 | 1110.78 | 1110.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 12:15:00 | 1133.60 | 1111.01 | 1110.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 1237.30 | 1260.94 | 1230.37 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-05 09:15:00 | 1286.00 | 1251.07 | 1234.78 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-23 15:15:00 | 1253.90 | 1272.88 | 1254.55 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 1221.40 | 1299.88 | 1300.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 1210.80 | 1299.00 | 1299.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 14:15:00 | 1250.90 | 1249.64 | 1270.19 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2026-04-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 14:15:00 | 1357.90 | 1286.29 | 1286.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 15:15:00 | 1364.00 | 1287.07 | 1286.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 10:15:00 | 1301.30 | 1311.89 | 1300.57 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-08-28 09:15:00 | 1174.50 | 2024-09-02 13:15:00 | 1191.40 | EXIT_EMA400 | -16.90 |
| SELL | 2024-10-28 09:15:00 | 1183.60 | 2024-10-29 09:15:00 | 1160.89 | TARGET | 22.71 |
| SELL | 2024-10-29 15:15:00 | 1183.35 | 2024-10-31 13:15:00 | 1162.45 | TARGET | 20.90 |
| SELL | 2024-10-22 12:15:00 | 1181.00 | 2024-11-04 11:15:00 | 1143.15 | TARGET | 37.85 |
| SELL | 2024-10-25 11:15:00 | 1171.60 | 2024-12-05 12:15:00 | 1168.90 | EXIT_EMA400 | 2.70 |
| SELL | 2024-11-12 12:15:00 | 1161.40 | 2024-12-05 12:15:00 | 1168.90 | EXIT_EMA400 | -7.50 |
| SELL | 2024-12-04 13:15:00 | 1156.10 | 2024-12-05 12:15:00 | 1168.90 | EXIT_EMA400 | -12.80 |
| BUY | 2025-06-06 10:15:00 | 1192.40 | 2025-07-01 09:15:00 | 1174.60 | EXIT_EMA400 | -17.80 |
| BUY | 2026-01-05 09:15:00 | 1286.00 | 2026-01-23 15:15:00 | 1253.90 | EXIT_EMA400 | -32.10 |
