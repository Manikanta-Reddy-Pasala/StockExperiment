# eClerx Services Ltd. (ECLERX.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1430.00
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
| EXIT | 2 |

## P&L

- **Trades closed:** 2
- **Trades open at end:** 1
- **Winners / losers:** 1 / 1
- **Target hits / EMA400 exits:** 1 / 1
- **Total realized P&L (per unit):** 399.12
- **Avg P&L per closed trade:** 199.56

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 09:15:00 | 1460.38 | 1652.75 | 1653.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 09:15:00 | 1430.50 | 1562.42 | 1592.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-03 14:15:00 | 1393.43 | 1393.01 | 1457.77 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-04 09:15:00 | 1343.70 | 1392.53 | 1456.89 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-05-13 14:15:00 | 1350.50 | 1284.87 | 1345.88 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 15:15:00 | 1701.00 | 1394.37 | 1393.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 1736.40 | 1468.64 | 1433.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 14:15:00 | 1687.50 | 1691.99 | 1594.63 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-23 09:15:00 | 1731.20 | 1692.58 | 1595.89 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 2040.25 | 2117.78 | 2025.28 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-29 11:15:00 | 2016.50 | 2115.95 | 2025.28 | Close below EMA400 |

### Cycle 3 — SELL (started 2026-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 13:15:00 | 1999.05 | 2254.17 | 2254.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 09:15:00 | 1958.75 | 2246.06 | 2250.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-16 09:15:00 | 1555.10 | 1553.74 | 1710.40 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-22 10:15:00 | 1484.90 | 1563.54 | 1694.57 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-04-04 09:15:00 | 1343.70 | 2025-05-13 14:15:00 | 1350.50 | EXIT_EMA400 | -6.80 |
| BUY | 2025-06-23 09:15:00 | 1731.20 | 2025-08-25 15:15:00 | 2137.12 | TARGET | 405.92 |
