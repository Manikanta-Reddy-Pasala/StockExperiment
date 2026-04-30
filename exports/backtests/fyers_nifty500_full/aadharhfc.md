# Aadhar Housing Finance Ltd. (AADHARHFC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-15 09:15:00 → 2026-04-30 15:15:00 (3395 bars)
- **Last close:** 490.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 5 |
| ENTRY1 | 3 |
| ENTRY2 | 3 |
| EXIT | 3 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 2 / 4
- **Total realized P&L (per unit):** 38.23
- **Avg P&L per closed trade:** 6.37

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 10:15:00 | 405.50 | 435.20 | 435.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-15 09:15:00 | 399.95 | 424.31 | 427.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 09:15:00 | 409.15 | 401.16 | 411.16 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-10 09:15:00 | 392.55 | 401.32 | 410.90 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 10:15:00 | 397.65 | 387.82 | 397.99 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-06 13:15:00 | 398.25 | 388.09 | 397.97 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-20 11:15:00 | 422.45 | 405.19 | 405.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 13:15:00 | 428.30 | 405.61 | 405.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 407.10 | 408.11 | 406.68 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-03-27 10:15:00 | 418.40 | 407.70 | 406.56 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 422.05 | 417.31 | 411.95 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-04-07 10:15:00 | 427.45 | 417.41 | 412.03 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 442.05 | 454.12 | 439.31 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-12 09:15:00 | 451.10 | 453.41 | 439.46 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 13:15:00 | 440.00 | 453.00 | 439.53 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-13 09:15:00 | 453.90 | 452.79 | 439.62 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 442.95 | 452.40 | 441.66 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-20 12:15:00 | 441.60 | 452.29 | 441.66 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 12:15:00 | 497.00 | 505.99 | 506.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 491.75 | 504.84 | 505.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 09:15:00 | 492.95 | 490.61 | 495.87 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-17 09:15:00 | 483.75 | 490.87 | 495.65 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-02 09:15:00 | 499.80 | 486.63 | 491.59 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 09:15:00 | 494.40 | 471.20 | 471.13 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-10 09:15:00 | 392.55 | 2025-03-06 13:15:00 | 398.25 | EXIT_EMA400 | -5.70 |
| BUY | 2025-03-27 10:15:00 | 418.40 | 2025-04-03 11:15:00 | 453.91 | TARGET | 35.51 |
| BUY | 2025-04-07 10:15:00 | 427.45 | 2025-04-16 11:15:00 | 473.72 | TARGET | 46.27 |
| BUY | 2025-05-12 09:15:00 | 451.10 | 2025-05-20 12:15:00 | 441.60 | EXIT_EMA400 | -9.50 |
| BUY | 2025-05-13 09:15:00 | 453.90 | 2025-05-20 12:15:00 | 441.60 | EXIT_EMA400 | -12.30 |
| SELL | 2025-12-17 09:15:00 | 483.75 | 2026-01-02 09:15:00 | 499.80 | EXIT_EMA400 | -16.05 |
