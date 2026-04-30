# Apar Industries Ltd. (APARINDS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 12331.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 1 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** -1100.00
- **Avg P&L per closed trade:** -275.00

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 15:15:00 | 7179.00 | 9892.51 | 9893.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-29 09:15:00 | 7007.55 | 9863.80 | 9879.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-24 09:15:00 | 5507.50 | 5486.09 | 6239.68 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 10:15:00 | 7567.50 | 6394.40 | 6393.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 11:15:00 | 7645.50 | 6406.85 | 6400.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 14:15:00 | 8654.00 | 8663.04 | 8167.72 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-29 12:15:00 | 9330.00 | 8668.24 | 8182.55 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-19 10:15:00 | 8420.00 | 8745.75 | 8427.70 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-09-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 15:15:00 | 7822.00 | 8251.12 | 8252.47 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 13:15:00 | 8551.00 | 8254.05 | 8253.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 10:15:00 | 8595.50 | 8264.56 | 8258.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 11:15:00 | 8486.50 | 8509.71 | 8404.32 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-16 09:15:00 | 8655.50 | 8443.51 | 8399.83 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 8655.50 | 8443.51 | 8399.83 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-16 13:15:00 | 8883.00 | 8458.30 | 8408.16 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-11-03 10:15:00 | 8544.00 | 8687.21 | 8552.98 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 11:15:00 | 8350.00 | 8726.06 | 8727.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-02 13:15:00 | 8253.00 | 8717.45 | 8723.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 10:15:00 | 7925.50 | 7851.72 | 8181.92 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2026-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 13:15:00 | 9523.00 | 8424.73 | 8422.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 09:15:00 | 9564.50 | 8456.49 | 8438.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 10:15:00 | 9782.00 | 9811.49 | 9325.82 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-25 09:15:00 | 9950.00 | 9653.91 | 9369.50 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-04-02 09:15:00 | 9332.00 | 9741.83 | 9454.40 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-07-29 12:15:00 | 9330.00 | 2025-08-19 10:15:00 | 8420.00 | EXIT_EMA400 | -910.00 |
| BUY | 2025-10-16 09:15:00 | 8655.50 | 2025-10-29 10:15:00 | 9422.50 | TARGET | 767.00 |
| BUY | 2025-10-16 13:15:00 | 8883.00 | 2025-11-03 10:15:00 | 8544.00 | EXIT_EMA400 | -339.00 |
| BUY | 2026-03-25 09:15:00 | 9950.00 | 2026-04-02 09:15:00 | 9332.00 | EXIT_EMA400 | -618.00 |
