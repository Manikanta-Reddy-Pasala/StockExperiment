# Mahanagar Gas Ltd. (MGL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1133.15
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 8 |
| ENTRY1 | 4 |
| ENTRY2 | 5 |
| EXIT | 4 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 3 / 6
- **Target hits / EMA400 exits:** 3 / 6
- **Total realized P&L (per unit):** 85.11
- **Avg P&L per closed trade:** 9.46

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 10:15:00 | 1563.95 | 1775.76 | 1776.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 11:15:00 | 1562.35 | 1773.63 | 1775.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 10:15:00 | 1285.45 | 1282.78 | 1378.52 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-07 09:15:00 | 1263.70 | 1285.42 | 1364.97 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 1301.05 | 1282.64 | 1328.87 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-01-29 14:15:00 | 1270.25 | 1282.36 | 1327.59 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 1317.85 | 1282.49 | 1327.20 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-31 09:15:00 | 1344.00 | 1284.51 | 1326.69 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-03-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-26 12:15:00 | 1366.20 | 1318.69 | 1318.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 09:15:00 | 1387.90 | 1323.07 | 1320.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 13:15:00 | 1329.85 | 1338.10 | 1329.31 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-16 11:15:00 | 1250.10 | 1321.89 | 1322.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-17 09:15:00 | 1245.20 | 1318.48 | 1320.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-22 09:15:00 | 1332.50 | 1312.65 | 1317.15 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-25 09:15:00 | 1294.30 | 1315.32 | 1318.12 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 1294.30 | 1315.32 | 1318.12 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-04-25 10:15:00 | 1278.40 | 1314.95 | 1317.92 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-04-28 09:15:00 | 1321.00 | 1313.48 | 1317.09 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 14:15:00 | 1360.00 | 1320.22 | 1320.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-02 15:15:00 | 1360.70 | 1320.62 | 1320.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 12:15:00 | 1330.60 | 1339.45 | 1330.66 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-09 13:15:00 | 1362.40 | 1339.68 | 1330.82 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 13:15:00 | 1362.40 | 1339.68 | 1330.82 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-09 15:15:00 | 1364.00 | 1340.15 | 1331.15 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 13:15:00 | 1346.10 | 1361.43 | 1345.93 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-23 12:15:00 | 1345.70 | 1360.83 | 1346.09 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 11:15:00 | 1285.60 | 1407.91 | 1408.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 13:15:00 | 1274.70 | 1357.25 | 1377.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 1331.60 | 1316.51 | 1345.11 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-24 10:15:00 | 1288.30 | 1323.10 | 1343.44 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 1302.70 | 1303.03 | 1326.99 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-08 13:15:00 | 1280.70 | 1302.88 | 1325.63 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 1315.80 | 1300.70 | 1320.43 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-16 13:15:00 | 1305.90 | 1301.24 | 1320.31 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 1117.20 | 1084.13 | 1124.19 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-06 11:15:00 | 1128.50 | 1084.57 | 1124.21 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 14:15:00 | 1150.90 | 1081.30 | 1080.96 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-07 09:15:00 | 1263.70 | 2025-01-31 09:15:00 | 1344.00 | EXIT_EMA400 | -80.30 |
| SELL | 2025-01-29 14:15:00 | 1270.25 | 2025-01-31 09:15:00 | 1344.00 | EXIT_EMA400 | -73.75 |
| SELL | 2025-04-25 09:15:00 | 1294.30 | 2025-04-28 09:15:00 | 1321.00 | EXIT_EMA400 | -26.70 |
| SELL | 2025-04-25 10:15:00 | 1278.40 | 2025-04-28 09:15:00 | 1321.00 | EXIT_EMA400 | -42.60 |
| BUY | 2025-05-09 13:15:00 | 1362.40 | 2025-05-23 12:15:00 | 1345.70 | EXIT_EMA400 | -16.70 |
| BUY | 2025-05-09 15:15:00 | 1364.00 | 2025-05-23 12:15:00 | 1345.70 | EXIT_EMA400 | -18.30 |
| SELL | 2025-10-16 13:15:00 | 1305.90 | 2025-10-30 09:15:00 | 1262.67 | TARGET | 43.23 |
| SELL | 2025-10-08 13:15:00 | 1280.70 | 2025-12-08 11:15:00 | 1145.90 | TARGET | 134.80 |
| SELL | 2025-09-24 10:15:00 | 1288.30 | 2025-12-08 13:15:00 | 1122.87 | TARGET | 165.43 |
