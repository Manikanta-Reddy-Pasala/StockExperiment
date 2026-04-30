# Kalpataru Projects International Ltd. (KPIL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1247.20
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT3 | 7 |
| ENTRY1 | 4 |
| ENTRY2 | 6 |
| EXIT | 4 |

## P&L

- **Trades closed:** 10
- **Trades open at end:** 0
- **Winners / losers:** 2 / 8
- **Target hits / EMA400 exits:** 1 / 9
- **Total realized P&L (per unit):** -190.67
- **Avg P&L per closed trade:** -19.07

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 11:15:00 | 1219.90 | 1307.67 | 1307.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 1194.75 | 1303.17 | 1305.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-01 17:15:00 | 1316.95 | 1282.00 | 1293.23 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-01 18:15:00 | 1286.70 | 1282.04 | 1293.19 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 1286.70 | 1282.04 | 1293.19 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-04 09:15:00 | 1263.35 | 1281.86 | 1293.04 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-11-07 11:15:00 | 1295.00 | 1276.42 | 1288.94 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 09:15:00 | 1300.50 | 1256.26 | 1256.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 11:15:00 | 1305.00 | 1257.20 | 1256.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 1258.05 | 1268.77 | 1263.13 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 11:15:00 | 1170.10 | 1258.10 | 1258.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 12:15:00 | 1166.40 | 1257.19 | 1257.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 14:15:00 | 937.60 | 934.55 | 1007.60 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-07 09:15:00 | 860.35 | 954.45 | 997.67 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-21 13:15:00 | 977.85 | 937.83 | 977.69 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 10:15:00 | 1126.45 | 988.03 | 987.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 11:15:00 | 1132.15 | 1023.90 | 1007.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 1181.00 | 1183.96 | 1148.28 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-08 09:15:00 | 1196.00 | 1156.12 | 1142.72 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 1231.60 | 1253.81 | 1227.57 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-09-30 09:15:00 | 1248.50 | 1253.57 | 1227.72 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1244.70 | 1252.44 | 1231.95 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-09 15:15:00 | 1258.00 | 1252.53 | 1232.60 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1242.00 | 1252.52 | 1233.38 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-13 13:15:00 | 1253.30 | 1252.28 | 1233.63 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-10-27 10:15:00 | 1237.50 | 1256.55 | 1240.62 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 13:15:00 | 1200.60 | 1243.05 | 1243.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 15:15:00 | 1199.40 | 1242.19 | 1242.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 11:15:00 | 1205.00 | 1189.03 | 1208.37 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-29 11:15:00 | 1181.90 | 1189.96 | 1207.55 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 1202.40 | 1188.91 | 1205.57 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-12-31 15:15:00 | 1195.50 | 1188.98 | 1205.52 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 1191.10 | 1189.00 | 1205.45 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-01 11:15:00 | 1188.70 | 1189.05 | 1205.31 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 13:15:00 | 1203.10 | 1189.30 | 1205.27 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-01-02 09:15:00 | 1207.10 | 1189.71 | 1205.24 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 10:15:00 | 1228.10 | 1133.11 | 1132.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 13:15:00 | 1243.00 | 1136.30 | 1134.50 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-01 18:15:00 | 1286.70 | 2024-11-04 09:15:00 | 1267.22 | TARGET | 19.48 |
| SELL | 2024-11-04 09:15:00 | 1263.35 | 2024-11-07 11:15:00 | 1295.00 | EXIT_EMA400 | -31.65 |
| SELL | 2025-04-07 09:15:00 | 860.35 | 2025-04-21 13:15:00 | 977.85 | EXIT_EMA400 | -117.50 |
| BUY | 2025-08-08 09:15:00 | 1196.00 | 2025-10-27 10:15:00 | 1237.50 | EXIT_EMA400 | 41.50 |
| BUY | 2025-09-30 09:15:00 | 1248.50 | 2025-10-27 10:15:00 | 1237.50 | EXIT_EMA400 | -11.00 |
| BUY | 2025-10-09 15:15:00 | 1258.00 | 2025-10-27 10:15:00 | 1237.50 | EXIT_EMA400 | -20.50 |
| BUY | 2025-10-13 13:15:00 | 1253.30 | 2025-10-27 10:15:00 | 1237.50 | EXIT_EMA400 | -15.80 |
| SELL | 2025-12-29 11:15:00 | 1181.90 | 2026-01-02 09:15:00 | 1207.10 | EXIT_EMA400 | -25.20 |
| SELL | 2025-12-31 15:15:00 | 1195.50 | 2026-01-02 09:15:00 | 1207.10 | EXIT_EMA400 | -11.60 |
| SELL | 2026-01-01 11:15:00 | 1188.70 | 2026-01-02 09:15:00 | 1207.10 | EXIT_EMA400 | -18.40 |
