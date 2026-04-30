# Action Construction Equipment Ltd. (ACE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 887.65
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 0 / 6
- **Target hits / EMA400 exits:** 0 / 6
- **Total realized P&L (per unit):** -262.25
- **Avg P&L per closed trade:** -43.71

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 13:15:00 | 1347.30 | 1424.62 | 1424.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 14:15:00 | 1342.95 | 1423.80 | 1424.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-16 13:15:00 | 1315.40 | 1292.52 | 1329.53 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-07 14:15:00 | 1284.00 | 1347.28 | 1349.74 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 1346.85 | 1342.25 | 1347.05 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-10-09 10:15:00 | 1348.40 | 1342.31 | 1347.06 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-17 13:15:00 | 1387.60 | 1350.50 | 1350.40 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 11:15:00 | 1271.65 | 1350.61 | 1350.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 1245.75 | 1341.59 | 1345.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 14:15:00 | 1347.45 | 1310.35 | 1328.13 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-04 10:15:00 | 1287.90 | 1310.59 | 1327.99 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 1320.00 | 1309.56 | 1326.36 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-07 09:15:00 | 1334.25 | 1310.47 | 1326.25 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 15:15:00 | 1423.95 | 1319.14 | 1319.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 09:15:00 | 1426.20 | 1320.21 | 1319.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-08 12:15:00 | 1437.30 | 1442.30 | 1398.86 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 13:15:00 | 1248.60 | 1370.08 | 1370.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 15:15:00 | 1238.95 | 1356.35 | 1363.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 11:15:00 | 1180.60 | 1175.89 | 1235.18 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-05-07 09:15:00 | 1143.40 | 1212.92 | 1223.85 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-05-08 11:15:00 | 1231.00 | 1210.38 | 1222.06 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 11:15:00 | 1278.00 | 1227.53 | 1227.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 13:15:00 | 1286.80 | 1228.61 | 1227.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 11:15:00 | 1246.00 | 1247.89 | 1239.92 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-09 09:15:00 | 1264.60 | 1247.87 | 1240.11 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 1242.90 | 1249.55 | 1241.85 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-12 13:15:00 | 1227.00 | 1249.29 | 1241.80 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 11:15:00 | 1179.00 | 1235.00 | 1235.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 12:15:00 | 1172.00 | 1234.37 | 1234.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-25 09:15:00 | 1224.10 | 1223.60 | 1229.11 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-25 11:15:00 | 1212.40 | 1223.42 | 1228.96 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 1225.70 | 1222.96 | 1228.40 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-06-27 13:15:00 | 1219.70 | 1222.88 | 1228.25 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-06-30 09:15:00 | 1229.20 | 1222.75 | 1228.11 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 15:15:00 | 905.90 | 880.08 | 880.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 907.90 | 880.36 | 880.21 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-07 14:15:00 | 1284.00 | 2024-10-09 10:15:00 | 1348.40 | EXIT_EMA400 | -64.40 |
| SELL | 2024-11-04 10:15:00 | 1287.90 | 2024-11-07 09:15:00 | 1334.25 | EXIT_EMA400 | -46.35 |
| SELL | 2025-05-07 09:15:00 | 1143.40 | 2025-05-08 11:15:00 | 1231.00 | EXIT_EMA400 | -87.60 |
| BUY | 2025-06-09 09:15:00 | 1264.60 | 2025-06-12 13:15:00 | 1227.00 | EXIT_EMA400 | -37.60 |
| SELL | 2025-06-25 11:15:00 | 1212.40 | 2025-06-30 09:15:00 | 1229.20 | EXIT_EMA400 | -16.80 |
| SELL | 2025-06-27 13:15:00 | 1219.70 | 2025-06-30 09:15:00 | 1229.20 | EXIT_EMA400 | -9.50 |
