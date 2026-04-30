# Ipca Laboratories Ltd. (IPCALAB.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1527.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 2 |
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 0 / 6
- **Total realized P&L (per unit):** -218.50
- **Avg P&L per closed trade:** -36.42

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-29 12:15:00 | 1414.65 | 1564.24 | 1564.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-12 09:15:00 | 1393.00 | 1506.64 | 1530.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 13:15:00 | 1515.30 | 1500.14 | 1525.62 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-14 09:15:00 | 1454.80 | 1499.45 | 1524.89 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 1416.75 | 1392.59 | 1439.70 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-24 12:15:00 | 1440.90 | 1393.85 | 1439.64 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 12:15:00 | 1495.50 | 1419.06 | 1419.06 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 10:15:00 | 1376.50 | 1419.35 | 1419.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-09 15:15:00 | 1368.80 | 1412.30 | 1415.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 09:15:00 | 1416.60 | 1407.52 | 1413.08 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-12 13:15:00 | 1381.40 | 1407.18 | 1412.80 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 1383.00 | 1389.69 | 1402.08 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-06-24 14:15:00 | 1332.30 | 1383.96 | 1398.35 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 1387.80 | 1375.70 | 1391.93 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-07-01 09:15:00 | 1370.90 | 1375.65 | 1391.82 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-07-03 09:15:00 | 1413.60 | 1375.40 | 1390.59 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 10:15:00 | 1444.70 | 1402.50 | 1402.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 11:15:00 | 1450.60 | 1402.98 | 1402.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 09:15:00 | 1444.20 | 1454.87 | 1434.02 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 09:15:00 | 1371.70 | 1420.89 | 1421.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 12:15:00 | 1357.20 | 1411.30 | 1415.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 12:15:00 | 1417.80 | 1408.85 | 1414.57 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-25 11:15:00 | 1392.80 | 1408.62 | 1414.29 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-08-26 14:15:00 | 1435.00 | 1407.12 | 1413.24 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 13:15:00 | 1441.10 | 1348.96 | 1348.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-21 10:15:00 | 1448.00 | 1352.50 | 1350.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 12:15:00 | 1399.00 | 1414.53 | 1391.28 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-18 15:15:00 | 1430.00 | 1412.64 | 1391.44 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-12-30 10:15:00 | 1396.00 | 1415.85 | 1397.44 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-14 09:15:00 | 1454.80 | 2025-03-24 12:15:00 | 1440.90 | EXIT_EMA400 | 13.90 |
| SELL | 2025-06-12 13:15:00 | 1381.40 | 2025-07-03 09:15:00 | 1413.60 | EXIT_EMA400 | -32.20 |
| SELL | 2025-06-24 14:15:00 | 1332.30 | 2025-07-03 09:15:00 | 1413.60 | EXIT_EMA400 | -81.30 |
| SELL | 2025-07-01 09:15:00 | 1370.90 | 2025-07-03 09:15:00 | 1413.60 | EXIT_EMA400 | -42.70 |
| SELL | 2025-08-25 11:15:00 | 1392.80 | 2025-08-26 14:15:00 | 1435.00 | EXIT_EMA400 | -42.20 |
| BUY | 2025-12-18 15:15:00 | 1430.00 | 2025-12-30 10:15:00 | 1396.00 | EXIT_EMA400 | -34.00 |
