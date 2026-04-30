# DCM Shriram Ltd. (DCMSHRIRAM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1209.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT3 | 5 |
| ENTRY1 | 4 |
| ENTRY2 | 3 |
| EXIT | 4 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 5 / 2
- **Target hits / EMA400 exits:** 5 / 2
- **Total realized P&L (per unit):** 442.23
- **Avg P&L per closed trade:** 63.18

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-08 10:15:00 | 996.00 | 1072.90 | 1073.19 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-07 15:15:00 | 1273.25 | 1067.56 | 1066.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-08 09:15:00 | 1288.55 | 1069.76 | 1068.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 09:15:00 | 1153.70 | 1169.96 | 1127.20 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 15:15:00 | 1095.00 | 1113.96 | 1113.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 09:15:00 | 1079.00 | 1113.61 | 1113.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 14:15:00 | 1149.80 | 1109.33 | 1111.46 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-01-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 13:15:00 | 1136.45 | 1113.55 | 1113.47 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 12:15:00 | 1073.70 | 1113.20 | 1113.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 1070.95 | 1112.78 | 1113.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-17 14:15:00 | 1104.95 | 1088.40 | 1098.94 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 09:15:00 | 1176.80 | 1106.43 | 1106.42 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 14:15:00 | 1074.05 | 1107.69 | 1107.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 13:15:00 | 1050.10 | 1105.24 | 1106.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 11:15:00 | 1032.35 | 1030.99 | 1060.97 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-05 12:15:00 | 1022.00 | 1030.90 | 1060.77 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 1056.20 | 1031.76 | 1059.62 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-10 09:15:00 | 1032.80 | 1032.64 | 1059.11 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-03-20 13:15:00 | 1059.95 | 1024.86 | 1048.48 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 1072.90 | 1047.50 | 1047.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 10:15:00 | 1081.10 | 1047.83 | 1047.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 14:15:00 | 1048.90 | 1049.67 | 1048.55 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-05 10:15:00 | 1063.60 | 1049.91 | 1048.69 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 1082.20 | 1050.56 | 1049.06 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-10 10:15:00 | 1093.00 | 1053.80 | 1050.84 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 12:15:00 | 1287.70 | 1348.20 | 1286.17 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-14 13:15:00 | 1278.90 | 1347.51 | 1286.14 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 13:15:00 | 1195.60 | 1263.71 | 1263.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 12:15:00 | 1181.60 | 1259.54 | 1261.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 15:15:00 | 1211.00 | 1209.10 | 1229.44 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-16 09:15:00 | 1193.60 | 1208.94 | 1229.26 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 1221.10 | 1208.88 | 1228.63 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-17 11:15:00 | 1240.10 | 1209.42 | 1228.61 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 14:15:00 | 1275.50 | 1241.05 | 1241.00 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2025-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 12:15:00 | 1211.60 | 1241.24 | 1241.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 1197.00 | 1239.92 | 1240.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 10:15:00 | 1242.40 | 1229.36 | 1235.04 | EMA200 retest candle locked |

### Cycle 12 — BUY (started 2025-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 12:15:00 | 1259.40 | 1233.86 | 1233.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 09:15:00 | 1261.10 | 1236.90 | 1235.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 11:15:00 | 1237.20 | 1241.32 | 1237.92 | EMA200 retest candle locked |

### Cycle 13 — SELL (started 2026-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 12:15:00 | 1180.00 | 1234.75 | 1234.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 14:15:00 | 1162.80 | 1220.48 | 1227.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 12:15:00 | 1179.20 | 1178.73 | 1201.83 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-01 14:15:00 | 1151.50 | 1178.25 | 1199.82 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1173.40 | 1175.89 | 1197.65 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-04 09:15:00 | 1152.80 | 1175.42 | 1196.67 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-03-25 10:15:00 | 1116.00 | 1060.92 | 1100.00 | Close above EMA400 |

### Cycle 14 — BUY (started 2026-04-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 14:15:00 | 1200.00 | 1119.85 | 1119.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 09:15:00 | 1212.00 | 1149.65 | 1136.51 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-03-10 09:15:00 | 1032.80 | 2025-03-13 09:15:00 | 953.88 | TARGET | 78.92 |
| SELL | 2025-03-05 12:15:00 | 1022.00 | 2025-03-20 13:15:00 | 1059.95 | EXIT_EMA400 | -37.95 |
| BUY | 2025-06-05 10:15:00 | 1063.60 | 2025-06-11 09:15:00 | 1108.32 | TARGET | 44.72 |
| BUY | 2025-06-10 10:15:00 | 1093.00 | 2025-06-30 09:15:00 | 1219.48 | TARGET | 126.48 |
| SELL | 2025-10-16 09:15:00 | 1193.60 | 2025-10-17 11:15:00 | 1240.10 | EXIT_EMA400 | -46.50 |
| SELL | 2026-02-01 14:15:00 | 1151.50 | 2026-03-02 09:15:00 | 1006.55 | TARGET | 144.95 |
| SELL | 2026-02-04 09:15:00 | 1152.80 | 2026-03-02 09:15:00 | 1021.19 | TARGET | 131.61 |
