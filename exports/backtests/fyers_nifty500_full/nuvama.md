# Nuvama Wealth Management Ltd. (NUVAMA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1334.40
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT3 | 5 |
| ENTRY1 | 4 |
| ENTRY2 | 3 |
| EXIT | 4 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 4 / 3
- **Target hits / EMA400 exits:** 3 / 4
- **Total realized P&L (per unit):** 353.75
- **Avg P&L per closed trade:** 50.54

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-16 10:15:00 | 1229.22 | 1344.87 | 1345.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-16 11:15:00 | 1228.20 | 1343.71 | 1344.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 14:15:00 | 1198.85 | 1189.52 | 1248.41 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-06 09:15:00 | 1178.04 | 1189.47 | 1247.80 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 12:15:00 | 1137.88 | 1091.78 | 1143.05 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-19 14:15:00 | 1123.89 | 1092.55 | 1142.93 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-03-20 09:15:00 | 1153.66 | 1093.47 | 1142.89 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 13:15:00 | 1260.20 | 1165.35 | 1164.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 09:15:00 | 1333.00 | 1191.13 | 1180.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 14:15:00 | 1394.20 | 1399.00 | 1331.33 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-23 10:15:00 | 1408.10 | 1398.61 | 1334.43 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 13:15:00 | 1450.00 | 1503.46 | 1448.05 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-28 14:15:00 | 1447.10 | 1502.90 | 1448.05 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 14:15:00 | 1382.00 | 1421.65 | 1421.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 12:15:00 | 1374.70 | 1418.24 | 1419.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 13:15:00 | 1294.00 | 1290.89 | 1331.40 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 09:15:00 | 1422.40 | 1357.21 | 1357.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 14:15:00 | 1450.00 | 1364.88 | 1361.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 11:15:00 | 1438.90 | 1440.45 | 1414.03 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-26 10:15:00 | 1467.00 | 1437.11 | 1413.96 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 1427.20 | 1448.12 | 1423.68 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-12-03 11:15:00 | 1437.50 | 1448.01 | 1423.75 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-12-03 13:15:00 | 1421.00 | 1447.61 | 1423.79 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 1270.00 | 1434.29 | 1434.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 13:15:00 | 1241.40 | 1429.18 | 1432.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 14:15:00 | 1392.00 | 1387.16 | 1407.63 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-05 09:15:00 | 1375.80 | 1387.05 | 1407.38 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 1378.20 | 1384.15 | 1403.30 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-11 09:15:00 | 1367.70 | 1383.92 | 1403.00 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1256.80 | 1200.95 | 1257.07 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-08 10:15:00 | 1285.20 | 1201.79 | 1257.21 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 1360.60 | 1290.46 | 1290.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 15:15:00 | 1362.70 | 1291.84 | 1291.12 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-06 09:15:00 | 1178.04 | 2025-03-20 09:15:00 | 1153.66 | EXIT_EMA400 | 24.38 |
| SELL | 2025-03-19 14:15:00 | 1123.89 | 2025-03-20 09:15:00 | 1153.66 | EXIT_EMA400 | -29.77 |
| BUY | 2025-06-23 10:15:00 | 1408.10 | 2025-06-27 13:15:00 | 1629.12 | TARGET | 221.02 |
| BUY | 2025-11-26 10:15:00 | 1467.00 | 2025-12-03 13:15:00 | 1421.00 | EXIT_EMA400 | -46.00 |
| BUY | 2025-12-03 11:15:00 | 1437.50 | 2025-12-03 13:15:00 | 1421.00 | EXIT_EMA400 | -16.50 |
| SELL | 2026-02-05 09:15:00 | 1375.80 | 2026-02-13 09:15:00 | 1281.07 | TARGET | 94.73 |
| SELL | 2026-02-11 09:15:00 | 1367.70 | 2026-02-16 09:15:00 | 1261.81 | TARGET | 105.89 |
