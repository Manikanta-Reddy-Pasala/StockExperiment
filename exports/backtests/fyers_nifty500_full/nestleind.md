# Nestle India Ltd. (NESTLEIND.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1463.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 0 / 5
- **Target hits / EMA400 exits:** 0 / 5
- **Total realized P&L (per unit):** -120.43
- **Avg P&L per closed trade:** -24.09

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 11:15:00 | 1244.00 | 1262.99 | 1263.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 09:15:00 | 1237.47 | 1261.20 | 1262.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 14:15:00 | 1263.65 | 1256.80 | 1259.65 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-19 14:15:00 | 1250.97 | 1256.89 | 1259.59 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 14:15:00 | 1250.97 | 1256.89 | 1259.59 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-08-20 09:15:00 | 1260.50 | 1256.88 | 1259.56 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 14:15:00 | 1274.50 | 1260.72 | 1260.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-17 09:15:00 | 1280.75 | 1261.06 | 1260.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 13:15:00 | 1307.10 | 1308.16 | 1289.15 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 11:15:00 | 1173.15 | 1276.59 | 1277.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 13:15:00 | 1167.50 | 1256.53 | 1266.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 10:15:00 | 1113.50 | 1106.45 | 1137.53 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-06 09:15:00 | 1101.50 | 1106.81 | 1136.80 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 1128.22 | 1106.27 | 1133.37 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-10 10:15:00 | 1134.25 | 1107.88 | 1133.26 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 10:15:00 | 1169.47 | 1116.34 | 1116.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 11:15:00 | 1176.85 | 1116.95 | 1116.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 12:15:00 | 1165.00 | 1165.27 | 1146.09 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-05 10:15:00 | 1174.20 | 1165.53 | 1146.70 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-05-09 09:15:00 | 1149.05 | 1166.46 | 1149.59 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 14:15:00 | 1115.85 | 1187.85 | 1187.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 1111.90 | 1169.11 | 1177.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 1161.20 | 1146.25 | 1163.55 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 1206.00 | 1171.03 | 1170.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 11:15:00 | 1216.20 | 1171.86 | 1171.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 14:15:00 | 1186.50 | 1187.13 | 1180.37 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-07 12:15:00 | 1190.70 | 1177.61 | 1176.75 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-10-07 14:15:00 | 1176.00 | 1177.67 | 1176.79 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 13:15:00 | 1234.50 | 1278.90 | 1279.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 1221.30 | 1277.41 | 1278.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1230.70 | 1222.56 | 1243.19 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-08 11:15:00 | 1210.30 | 1222.43 | 1242.92 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 13:15:00 | 1240.40 | 1222.74 | 1241.49 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-10 14:15:00 | 1248.60 | 1223.00 | 1241.52 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 09:15:00 | 1408.00 | 1253.92 | 1253.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 09:15:00 | 1416.30 | 1264.47 | 1259.24 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-08-19 14:15:00 | 1250.97 | 2024-08-20 09:15:00 | 1260.50 | EXIT_EMA400 | -9.53 |
| SELL | 2025-01-06 09:15:00 | 1101.50 | 2025-01-10 10:15:00 | 1134.25 | EXIT_EMA400 | -32.75 |
| BUY | 2025-05-05 10:15:00 | 1174.20 | 2025-05-09 09:15:00 | 1149.05 | EXIT_EMA400 | -25.15 |
| BUY | 2025-10-07 12:15:00 | 1190.70 | 2025-10-07 14:15:00 | 1176.00 | EXIT_EMA400 | -14.70 |
| SELL | 2026-04-08 11:15:00 | 1210.30 | 2026-04-10 14:15:00 | 1248.60 | EXIT_EMA400 | -38.30 |
