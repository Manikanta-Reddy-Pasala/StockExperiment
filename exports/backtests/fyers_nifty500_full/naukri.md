# Info Edge (India) Ltd. (NAUKRI.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 970.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 1 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 1 / 2
- **Target hits / EMA400 exits:** 1 / 2
- **Total realized P&L (per unit):** -20.57
- **Avg P&L per closed trade:** -6.86

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 11:15:00 | 1498.49 | 1628.78 | 1629.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 13:15:00 | 1491.00 | 1626.09 | 1628.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 15:15:00 | 1562.40 | 1561.85 | 1589.55 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-03 13:15:00 | 1549.31 | 1561.88 | 1588.88 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-05 11:15:00 | 1590.17 | 1560.55 | 1586.61 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 14:15:00 | 1490.50 | 1429.23 | 1428.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 12:15:00 | 1502.90 | 1435.79 | 1432.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 1452.00 | 1456.68 | 1444.76 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-13 11:15:00 | 1458.00 | 1456.59 | 1444.84 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 1467.40 | 1478.27 | 1461.47 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-01 12:15:00 | 1454.90 | 1477.89 | 1461.45 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 13:15:00 | 1400.70 | 1451.73 | 1451.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 10:15:00 | 1394.50 | 1449.67 | 1450.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 09:15:00 | 1455.90 | 1434.38 | 1442.43 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-28 12:15:00 | 1403.30 | 1438.37 | 1443.52 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-08-21 10:15:00 | 1422.50 | 1383.25 | 1406.04 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-03 13:15:00 | 1549.31 | 2025-02-05 11:15:00 | 1590.17 | EXIT_EMA400 | -40.86 |
| BUY | 2025-06-13 11:15:00 | 1458.00 | 2025-06-17 09:15:00 | 1497.49 | TARGET | 39.49 |
| SELL | 2025-07-28 12:15:00 | 1403.30 | 2025-08-21 10:15:00 | 1422.50 | EXIT_EMA400 | -19.20 |
