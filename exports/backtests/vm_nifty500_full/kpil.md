# Kalpataru Projects International Ltd. (KPIL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-04-30 15:30:00 (4976 bars)
- **Last close:** 1250.30
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT3 | 4 |
| ENTRY1 | 4 |
| ENTRY2 | 2 |
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** -76.78
- **Avg P&L per closed trade:** -12.80

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 15:15:00 | 1218.95 | 1304.14 | 1304.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 1194.75 | 1303.05 | 1303.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 14:15:00 | 1278.00 | 1275.87 | 1287.72 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-08 10:15:00 | 1255.00 | 1275.85 | 1287.14 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-05 12:15:00 | 1248.80 | 1197.61 | 1233.33 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 13:15:00 | 1286.95 | 1255.25 | 1255.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 09:15:00 | 1300.50 | 1256.32 | 1255.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 1258.05 | 1268.82 | 1262.83 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-01-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 12:15:00 | 1166.40 | 1257.27 | 1257.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 13:15:00 | 1147.45 | 1256.17 | 1257.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 14:15:00 | 937.40 | 935.35 | 1009.24 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-07 09:15:00 | 860.35 | 954.77 | 998.77 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 13:15:00 | 977.80 | 938.02 | 978.53 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-21 14:15:00 | 1009.15 | 938.72 | 978.69 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 10:15:00 | 1126.45 | 987.95 | 987.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 11:15:00 | 1132.15 | 1023.86 | 1007.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 1181.90 | 1184.07 | 1148.42 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-08 09:15:00 | 1196.00 | 1156.17 | 1142.81 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-29 15:15:00 | 1226.80 | 1253.59 | 1227.63 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 13:15:00 | 1200.60 | 1243.08 | 1243.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 15:15:00 | 1198.20 | 1242.20 | 1242.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 11:15:00 | 1205.00 | 1189.20 | 1208.49 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-29 11:15:00 | 1181.90 | 1190.14 | 1207.68 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 1202.40 | 1189.06 | 1205.68 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-12-31 15:15:00 | 1195.50 | 1189.13 | 1205.63 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 1190.30 | 1189.14 | 1205.56 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-01 11:15:00 | 1188.70 | 1189.19 | 1205.42 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 13:15:00 | 1203.10 | 1189.43 | 1205.38 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-01-02 09:15:00 | 1207.10 | 1189.84 | 1205.35 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 14:15:00 | 1244.80 | 1133.31 | 1133.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 10:15:00 | 1247.00 | 1136.58 | 1134.75 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-08 10:15:00 | 1255.00 | 2024-11-21 09:15:00 | 1158.58 | TARGET | 96.42 |
| SELL | 2025-04-07 09:15:00 | 860.35 | 2025-04-21 14:15:00 | 1009.15 | EXIT_EMA400 | -148.80 |
| BUY | 2025-08-08 09:15:00 | 1196.00 | 2025-09-29 15:15:00 | 1226.80 | EXIT_EMA400 | 30.80 |
| SELL | 2025-12-29 11:15:00 | 1181.90 | 2026-01-02 09:15:00 | 1207.10 | EXIT_EMA400 | -25.20 |
| SELL | 2025-12-31 15:15:00 | 1195.50 | 2026-01-02 09:15:00 | 1207.10 | EXIT_EMA400 | -11.60 |
| SELL | 2026-01-01 11:15:00 | 1188.70 | 2026-01-02 09:15:00 | 1207.10 | EXIT_EMA400 | -18.40 |
