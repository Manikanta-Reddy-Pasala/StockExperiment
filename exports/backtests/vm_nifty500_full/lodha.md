# Lodha Developers Ltd. (LODHA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (4998 bars)
- **Last close:** 897.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / EMA400 exits:** 3 / 3
- **Total realized P&L (per unit):** 340.19
- **Avg P&L per closed trade:** 56.70

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 15:15:00 | 1220.00 | 1364.25 | 1364.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 09:15:00 | 1210.40 | 1362.72 | 1363.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-16 09:15:00 | 1285.05 | 1246.87 | 1283.72 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-30 13:15:00 | 1240.00 | 1295.10 | 1300.30 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1205.05 | 1182.35 | 1222.56 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-04 10:15:00 | 1197.55 | 1182.51 | 1222.43 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 11:15:00 | 1213.80 | 1182.24 | 1219.42 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-06 14:15:00 | 1221.00 | 1183.30 | 1219.40 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 10:15:00 | 1296.00 | 1234.65 | 1234.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 10:15:00 | 1317.85 | 1238.90 | 1236.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 09:15:00 | 1358.00 | 1370.25 | 1322.37 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-31 14:15:00 | 1390.85 | 1370.55 | 1323.70 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 1346.30 | 1371.07 | 1330.53 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-07 11:15:00 | 1325.50 | 1370.29 | 1330.54 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 12:15:00 | 1174.00 | 1303.25 | 1303.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 09:15:00 | 1168.55 | 1291.82 | 1297.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 11:15:00 | 1230.80 | 1219.04 | 1254.17 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-31 14:15:00 | 1205.15 | 1218.75 | 1253.51 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-03 09:15:00 | 1255.05 | 1218.97 | 1253.27 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 11:15:00 | 1352.60 | 1196.93 | 1196.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-23 13:15:00 | 1357.70 | 1200.06 | 1198.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 1241.80 | 1258.87 | 1233.92 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 09:15:00 | 1292.30 | 1257.73 | 1234.20 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 1385.00 | 1430.22 | 1382.19 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-01 09:15:00 | 1372.80 | 1429.65 | 1382.15 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 12:15:00 | 1244.30 | 1373.39 | 1373.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 13:15:00 | 1236.00 | 1372.03 | 1373.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 13:15:00 | 1299.80 | 1285.44 | 1318.46 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-22 10:15:00 | 1276.10 | 1286.66 | 1317.33 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-31 09:15:00 | 1202.80 | 1175.66 | 1202.00 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-30 13:15:00 | 1240.00 | 2024-10-23 09:15:00 | 1059.09 | TARGET | 180.91 |
| SELL | 2024-11-04 10:15:00 | 1197.55 | 2024-11-06 14:15:00 | 1221.00 | EXIT_EMA400 | -23.45 |
| BUY | 2024-12-31 14:15:00 | 1390.85 | 2025-01-07 11:15:00 | 1325.50 | EXIT_EMA400 | -65.35 |
| SELL | 2025-01-31 14:15:00 | 1205.15 | 2025-02-03 09:15:00 | 1255.05 | EXIT_EMA400 | -49.90 |
| BUY | 2025-05-12 09:15:00 | 1292.30 | 2025-05-26 09:15:00 | 1466.59 | TARGET | 174.29 |
| SELL | 2025-08-22 10:15:00 | 1276.10 | 2025-09-25 13:15:00 | 1152.41 | TARGET | 123.69 |
