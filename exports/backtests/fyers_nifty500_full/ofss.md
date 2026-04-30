# Oracle Financial Services Software Ltd. (OFSS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 9736.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
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
- **Total realized P&L (per unit):** -487.00
- **Avg P&L per closed trade:** -81.17

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 10:15:00 | 10061.00 | 11738.16 | 11745.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 14:15:00 | 10007.55 | 11672.07 | 11712.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-16 10:15:00 | 7830.00 | 7825.64 | 8469.10 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2025-06-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 10:15:00 | 9374.00 | 8523.82 | 8523.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 11:15:00 | 9436.00 | 8532.89 | 8527.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 13:15:00 | 9010.00 | 9024.14 | 8839.43 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-27 15:15:00 | 9050.50 | 9024.16 | 8841.28 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 8894.50 | 9017.27 | 8851.80 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-02 14:15:00 | 8953.00 | 9013.52 | 8853.18 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-07-10 09:15:00 | 8852.50 | 9010.92 | 8879.19 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 8471.50 | 8811.13 | 8811.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 15:15:00 | 8453.50 | 8807.57 | 8809.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 10:15:00 | 8670.00 | 8663.16 | 8724.72 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-14 11:15:00 | 8615.00 | 8662.68 | 8724.17 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-08-20 11:15:00 | 8774.50 | 8651.33 | 8712.00 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 9141.50 | 8699.90 | 8699.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 14:15:00 | 9185.00 | 8744.12 | 8722.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 09:15:00 | 8742.50 | 8833.36 | 8775.29 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-03 12:15:00 | 8928.00 | 8762.39 | 8744.98 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-10-15 11:15:00 | 8809.00 | 8922.83 | 8840.81 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 09:15:00 | 8589.00 | 8784.23 | 8785.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 10:15:00 | 8558.50 | 8781.99 | 8783.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 8489.00 | 8485.57 | 8602.71 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-21 09:15:00 | 8272.50 | 8478.87 | 8595.29 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 7959.50 | 7799.95 | 8009.63 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-20 13:15:00 | 7811.50 | 7815.98 | 8004.74 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 7985.00 | 7807.03 | 7990.94 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-01-23 10:15:00 | 7997.00 | 7815.52 | 7988.05 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 15:15:00 | 7980.00 | 7201.84 | 7200.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 09:15:00 | 8091.50 | 7210.69 | 7204.58 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-06-27 15:15:00 | 9050.50 | 2025-07-10 09:15:00 | 8852.50 | EXIT_EMA400 | -198.00 |
| BUY | 2025-07-02 14:15:00 | 8953.00 | 2025-07-10 09:15:00 | 8852.50 | EXIT_EMA400 | -100.50 |
| SELL | 2025-08-14 11:15:00 | 8615.00 | 2025-08-20 11:15:00 | 8774.50 | EXIT_EMA400 | -159.50 |
| BUY | 2025-10-03 12:15:00 | 8928.00 | 2025-10-15 11:15:00 | 8809.00 | EXIT_EMA400 | -119.00 |
| SELL | 2025-11-21 09:15:00 | 8272.50 | 2026-01-23 10:15:00 | 7997.00 | EXIT_EMA400 | 275.50 |
| SELL | 2026-01-20 13:15:00 | 7811.50 | 2026-01-23 10:15:00 | 7997.00 | EXIT_EMA400 | -185.50 |
