# Onesource Specialty Pharma Ltd. (ONESOURCE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2025-01-24 09:15:00 → 2026-04-30 15:30:00 (2162 bars)
- **Last close:** 1738.10
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 2 |
| ALERT3 | 1 |
| ENTRY1 | 2 |
| ENTRY2 | 1 |
| EXIT | 2 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 2 / 1
- **Target hits / EMA400 exits:** 2 / 1
- **Total realized P&L (per unit):** 230.07
- **Avg P&L per closed trade:** 76.69

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 12:15:00 | 1588.00 | 1552.39 | 1552.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 09:15:00 | 1630.00 | 1554.55 | 1553.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 10:15:00 | 1560.70 | 1572.69 | 1563.35 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-09 15:15:00 | 1619.00 | 1573.41 | 1563.95 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 1606.60 | 1581.14 | 1568.99 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-19 14:15:00 | 1625.00 | 1584.55 | 1571.86 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-08-01 09:15:00 | 1898.50 | 1979.61 | 1920.43 | Close below EMA400 |

### Cycle 2 — SELL (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 11:15:00 | 1867.80 | 1892.31 | 1892.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 12:15:00 | 1861.00 | 1890.64 | 1891.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 14:15:00 | 1839.50 | 1830.42 | 1854.63 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-07 09:15:00 | 1745.50 | 1832.59 | 1842.92 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-11-17 10:15:00 | 1840.00 | 1807.74 | 1827.23 | Close above EMA400 |

### Cycle 3 — BUY (started 2026-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 15:15:00 | 1611.00 | 1444.76 | 1444.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 09:15:00 | 1714.00 | 1447.44 | 1445.45 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-05-09 15:15:00 | 1619.00 | 2025-05-27 09:15:00 | 1784.16 | TARGET | 165.16 |
| BUY | 2025-05-19 14:15:00 | 1625.00 | 2025-05-27 09:15:00 | 1784.41 | TARGET | 159.41 |
| SELL | 2025-11-07 09:15:00 | 1745.50 | 2025-11-17 10:15:00 | 1840.00 | EXIT_EMA400 | -94.50 |
