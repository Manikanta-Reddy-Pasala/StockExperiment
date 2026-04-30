# Onesource Specialty Pharma Ltd. (ONESOURCE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2025-01-24 09:15:00 → 2026-04-30 15:15:00 (2179 bars)
- **Last close:** 1751.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 2 |
| ALERT3 | 2 |
| ENTRY1 | 2 |
| ENTRY2 | 1 |
| EXIT | 2 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 3 / 0
- **Target hits / EMA400 exits:** 3 / 0
- **Total realized P&L (per unit):** 442.09
- **Avg P&L per closed trade:** 147.36

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 10:15:00 | 1628.10 | 1550.84 | 1550.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 14:15:00 | 1657.10 | 1557.74 | 1554.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 10:15:00 | 1560.70 | 1572.22 | 1562.27 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-09 15:15:00 | 1619.00 | 1572.96 | 1562.89 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 1606.60 | 1580.76 | 1568.04 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-19 14:15:00 | 1625.00 | 1584.13 | 1570.94 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-08-01 09:15:00 | 1900.00 | 1979.40 | 1920.14 | Close below EMA400 |

### Cycle 2 — SELL (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 11:15:00 | 1867.80 | 1892.10 | 1892.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 12:15:00 | 1861.00 | 1890.45 | 1891.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 15:15:00 | 1870.00 | 1868.75 | 1878.38 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-19 09:15:00 | 1841.00 | 1868.47 | 1878.19 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 1839.50 | 1830.27 | 1854.46 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-06 15:15:00 | 1889.90 | 1825.22 | 1849.25 | Close above EMA400 |

### Cycle 3 — BUY (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 10:15:00 | 1571.90 | 1439.24 | 1439.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 11:15:00 | 1592.00 | 1440.76 | 1439.97 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-05-09 15:15:00 | 1619.00 | 2025-05-27 09:15:00 | 1787.33 | TARGET | 168.33 |
| BUY | 2025-05-19 14:15:00 | 1625.00 | 2025-05-27 09:15:00 | 1787.18 | TARGET | 162.18 |
| SELL | 2025-09-19 09:15:00 | 1841.00 | 2025-09-25 09:15:00 | 1729.42 | TARGET | 111.58 |
