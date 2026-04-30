# Rainbow Childrens Medicare Ltd. (RAINBOW.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1251.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 4 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** 12.75
- **Avg P&L per closed trade:** 3.19

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 09:15:00 | 1341.65 | 1244.60 | 1244.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 10:15:00 | 1355.50 | 1245.70 | 1245.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-22 10:15:00 | 1365.00 | 1381.92 | 1339.47 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-24 10:15:00 | 1385.10 | 1380.58 | 1341.63 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 1536.35 | 1589.34 | 1532.31 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-12-23 09:15:00 | 1528.65 | 1588.23 | 1532.32 | Close below EMA400 |

### Cycle 2 — SELL (started 2025-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-23 09:15:00 | 1409.75 | 1518.57 | 1519.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 1369.05 | 1509.95 | 1514.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 11:15:00 | 1317.85 | 1312.62 | 1367.40 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-18 13:15:00 | 1303.15 | 1312.45 | 1366.77 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 14:15:00 | 1357.70 | 1311.79 | 1359.00 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-24 15:15:00 | 1360.00 | 1312.27 | 1359.01 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-04-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 13:15:00 | 1544.70 | 1381.80 | 1381.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 14:15:00 | 1573.00 | 1383.70 | 1382.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-25 09:15:00 | 1398.60 | 1415.38 | 1400.84 | EMA200 retest candle locked |

### Cycle 4 — SELL (started 2025-05-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 11:15:00 | 1320.50 | 1390.51 | 1390.84 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 09:15:00 | 1426.60 | 1385.73 | 1385.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-12 12:15:00 | 1445.80 | 1392.16 | 1389.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 14:15:00 | 1519.40 | 1519.95 | 1484.26 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-01 09:15:00 | 1528.50 | 1519.85 | 1485.78 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1493.00 | 1519.34 | 1487.83 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-05 11:15:00 | 1484.40 | 1518.73 | 1487.84 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 13:15:00 | 1450.30 | 1496.51 | 1496.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 09:15:00 | 1430.50 | 1494.93 | 1495.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 1378.30 | 1375.49 | 1412.80 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-06 14:15:00 | 1361.80 | 1376.52 | 1405.55 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 1368.00 | 1349.88 | 1369.43 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-12-11 14:15:00 | 1378.50 | 1350.17 | 1369.47 | Close above EMA400 |

### Cycle 7 — BUY (started 2026-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 11:15:00 | 1302.20 | 1213.02 | 1212.95 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-10-24 10:15:00 | 1385.10 | 2024-10-29 09:15:00 | 1515.50 | TARGET | 130.40 |
| SELL | 2025-03-18 13:15:00 | 1303.15 | 2025-03-24 15:15:00 | 1360.00 | EXIT_EMA400 | -56.85 |
| BUY | 2025-08-01 09:15:00 | 1528.50 | 2025-08-05 11:15:00 | 1484.40 | EXIT_EMA400 | -44.10 |
| SELL | 2025-11-06 14:15:00 | 1361.80 | 2025-12-11 14:15:00 | 1378.50 | EXIT_EMA400 | -16.70 |
