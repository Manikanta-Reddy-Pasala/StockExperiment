# Aadhar Housing Finance Ltd. (AADHARHFC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-15 09:15:00 → 2026-04-30 15:30:00 (3373 bars)
- **Last close:** 488.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT3 | 4 |
| ENTRY1 | 3 |
| ENTRY2 | 2 |
| EXIT | 3 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / EMA400 exits:** 2 / 3
- **Total realized P&L (per unit):** 14.97
- **Avg P&L per closed trade:** 2.99

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 10:15:00 | 405.50 | 435.15 | 435.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-15 09:15:00 | 399.95 | 424.31 | 427.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 09:15:00 | 409.90 | 401.58 | 411.70 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-10 09:15:00 | 392.55 | 401.71 | 411.41 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 10:15:00 | 398.05 | 387.97 | 398.30 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-06 14:15:00 | 399.60 | 388.36 | 398.29 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-20 12:15:00 | 424.15 | 405.53 | 405.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 13:15:00 | 428.30 | 405.75 | 405.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 407.10 | 408.22 | 406.91 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-03-27 10:15:00 | 418.40 | 407.80 | 406.78 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 442.05 | 454.20 | 439.45 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-09 11:15:00 | 444.60 | 453.97 | 439.48 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 13:15:00 | 440.00 | 453.08 | 439.66 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-13 09:15:00 | 453.90 | 452.86 | 439.75 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 442.95 | 452.45 | 441.76 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-20 12:15:00 | 441.60 | 452.34 | 441.76 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 12:15:00 | 497.00 | 505.99 | 506.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 491.75 | 504.84 | 505.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 09:15:00 | 492.95 | 490.61 | 495.87 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-17 09:15:00 | 483.70 | 490.85 | 495.64 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-02 09:15:00 | 499.60 | 486.58 | 491.56 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 12:15:00 | 487.55 | 471.25 | 471.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 13:15:00 | 492.65 | 471.46 | 471.31 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-10 09:15:00 | 392.55 | 2025-03-06 14:15:00 | 399.60 | EXIT_EMA400 | -7.05 |
| BUY | 2025-03-27 10:15:00 | 418.40 | 2025-04-03 11:15:00 | 453.25 | TARGET | 34.85 |
| BUY | 2025-05-09 11:15:00 | 444.60 | 2025-05-12 09:15:00 | 459.97 | TARGET | 15.37 |
| BUY | 2025-05-13 09:15:00 | 453.90 | 2025-05-20 12:15:00 | 441.60 | EXIT_EMA400 | -12.30 |
| SELL | 2025-12-17 09:15:00 | 483.70 | 2026-01-02 09:15:00 | 499.60 | EXIT_EMA400 | -15.90 |
