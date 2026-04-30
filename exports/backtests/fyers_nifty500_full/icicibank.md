# ICICI Bank Ltd. (ICICIBANK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1265.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 3 |
| ENTRY1 | 6 |
| ENTRY2 | 2 |
| EXIT | 6 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 2 / 6
- **Target hits / EMA400 exits:** 1 / 7
- **Total realized P&L (per unit):** -18.42
- **Avg P&L per closed trade:** -2.30

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 09:15:00 | 1237.35 | 1279.98 | 1280.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 09:15:00 | 1226.10 | 1275.07 | 1277.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 09:15:00 | 1256.25 | 1252.06 | 1264.12 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-21 09:15:00 | 1227.00 | 1253.04 | 1259.15 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-11 13:15:00 | 1247.15 | 1234.28 | 1245.85 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-03-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-21 11:15:00 | 1341.20 | 1254.85 | 1254.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-24 09:15:00 | 1358.20 | 1259.18 | 1256.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 1288.00 | 1295.39 | 1278.51 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-15 09:15:00 | 1347.30 | 1296.83 | 1281.42 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-17 09:15:00 | 1412.60 | 1428.68 | 1415.32 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 12:15:00 | 1391.00 | 1429.41 | 1429.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 1381.30 | 1416.09 | 1421.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 1389.80 | 1388.78 | 1401.92 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-21 14:15:00 | 1381.30 | 1393.54 | 1402.71 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-11-13 10:15:00 | 1385.30 | 1368.50 | 1383.91 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-01-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 13:15:00 | 1416.60 | 1377.36 | 1377.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 09:15:00 | 1422.70 | 1378.55 | 1377.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 09:15:00 | 1362.10 | 1386.14 | 1381.98 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 1347.60 | 1378.57 | 1378.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 14:15:00 | 1344.30 | 1378.23 | 1378.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 13:15:00 | 1378.00 | 1375.40 | 1376.90 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-30 10:15:00 | 1364.90 | 1375.48 | 1376.91 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 1364.90 | 1375.48 | 1376.91 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-30 11:15:00 | 1360.40 | 1375.33 | 1376.83 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-02-03 09:15:00 | 1390.10 | 1371.24 | 1374.55 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 12:15:00 | 1401.90 | 1377.60 | 1377.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 13:15:00 | 1403.70 | 1377.86 | 1377.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 1386.80 | 1391.26 | 1385.53 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-20 09:15:00 | 1398.10 | 1391.31 | 1385.61 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 1398.10 | 1391.31 | 1385.61 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-02-23 09:15:00 | 1406.70 | 1391.73 | 1386.02 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 12:15:00 | 1390.00 | 1392.21 | 1386.55 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-02-24 14:15:00 | 1384.70 | 1392.13 | 1386.56 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 14:15:00 | 1312.00 | 1383.02 | 1383.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 1271.90 | 1381.25 | 1382.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1302.40 | 1281.40 | 1317.76 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-09 11:15:00 | 1290.30 | 1283.30 | 1317.13 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-10 09:15:00 | 1320.80 | 1283.69 | 1316.50 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-21 09:15:00 | 1227.00 | 2025-03-11 13:15:00 | 1247.15 | EXIT_EMA400 | -20.15 |
| BUY | 2025-04-15 09:15:00 | 1347.30 | 2025-07-17 09:15:00 | 1412.60 | EXIT_EMA400 | 65.30 |
| SELL | 2025-10-21 14:15:00 | 1381.30 | 2025-11-13 10:15:00 | 1385.30 | EXIT_EMA400 | -4.00 |
| SELL | 2026-01-30 10:15:00 | 1364.90 | 2026-02-01 15:15:00 | 1328.87 | TARGET | 36.03 |
| SELL | 2026-01-30 11:15:00 | 1360.40 | 2026-02-03 09:15:00 | 1390.10 | EXIT_EMA400 | -29.70 |
| BUY | 2026-02-20 09:15:00 | 1398.10 | 2026-02-24 14:15:00 | 1384.70 | EXIT_EMA400 | -13.40 |
| BUY | 2026-02-23 09:15:00 | 1406.70 | 2026-02-24 14:15:00 | 1384.70 | EXIT_EMA400 | -22.00 |
| SELL | 2026-04-09 11:15:00 | 1290.30 | 2026-04-10 09:15:00 | 1320.80 | EXIT_EMA400 | -30.50 |
