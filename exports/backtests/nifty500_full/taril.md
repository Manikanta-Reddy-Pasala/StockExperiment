# Transformers And Rectifiers (India) Ltd. (TARIL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-08-26 09:15:00 → 2026-04-30 15:15:00 (2889 bars)
- **Last close:** 334.20
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT3 | 8 |
| ENTRY1 | 5 |
| ENTRY2 | 7 |
| EXIT | 5 |

## P&L

- **Trades closed:** 12
- **Trades open at end:** 0
- **Winners / losers:** 3 / 9
- **Target hits / EMA400 exits:** 3 / 9
- **Total realized P&L (per unit):** -67.65
- **Avg P&L per closed trade:** -5.64

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-04 11:15:00 | 389.25 | 495.46 | 495.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-04 12:15:00 | 384.77 | 494.36 | 495.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 11:15:00 | 450.90 | 441.72 | 462.76 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-25 11:15:00 | 433.50 | 441.82 | 462.09 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 430.75 | 441.20 | 461.27 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-27 11:15:00 | 420.00 | 440.82 | 460.88 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 10:15:00 | 436.75 | 414.37 | 437.31 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-19 12:15:00 | 438.45 | 414.81 | 437.30 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 15:15:00 | 509.40 | 453.42 | 453.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 10:15:00 | 512.00 | 454.49 | 453.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-28 14:15:00 | 504.75 | 505.86 | 485.95 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-29 09:15:00 | 512.15 | 505.95 | 486.19 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 14:15:00 | 495.70 | 505.23 | 486.97 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-02 09:15:00 | 499.85 | 505.05 | 487.07 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 488.75 | 504.41 | 487.36 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-05 12:15:00 | 501.75 | 504.28 | 487.55 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-05-06 14:15:00 | 483.20 | 503.53 | 487.91 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-07-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 14:15:00 | 470.80 | 498.52 | 498.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 467.50 | 497.94 | 498.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 492.50 | 491.73 | 494.81 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 11:15:00 | 514.00 | 497.41 | 497.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 09:15:00 | 518.85 | 498.26 | 497.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 09:15:00 | 500.90 | 501.79 | 499.85 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-29 13:15:00 | 510.80 | 499.01 | 498.60 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 510.80 | 499.01 | 498.60 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-31 12:15:00 | 513.35 | 500.29 | 499.29 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 505.45 | 507.21 | 503.05 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-08-06 11:15:00 | 512.70 | 507.26 | 503.10 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-08-07 09:15:00 | 499.10 | 507.25 | 503.20 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 15:15:00 | 496.40 | 502.44 | 502.46 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 509.10 | 502.54 | 502.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 12:15:00 | 519.20 | 502.77 | 502.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 11:15:00 | 502.00 | 503.29 | 502.90 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-15 10:15:00 | 506.00 | 503.03 | 502.79 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 506.00 | 503.03 | 502.79 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-09-16 09:15:00 | 517.35 | 503.26 | 502.91 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-09-25 14:15:00 | 506.65 | 513.14 | 508.62 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 11:15:00 | 490.40 | 505.04 | 505.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 09:15:00 | 487.35 | 504.25 | 504.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 09:15:00 | 304.25 | 302.72 | 360.58 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-12 09:15:00 | 266.60 | 301.80 | 331.37 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 14:15:00 | 287.86 | 262.56 | 291.43 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-10 09:15:00 | 277.60 | 262.88 | 291.30 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-02-11 09:15:00 | 294.20 | 264.22 | 291.00 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 13:15:00 | 330.95 | 289.49 | 289.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 15:15:00 | 334.90 | 290.37 | 289.84 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-25 11:15:00 | 433.50 | 2025-03-19 12:15:00 | 438.45 | EXIT_EMA400 | -4.95 |
| SELL | 2025-02-27 11:15:00 | 420.00 | 2025-03-19 12:15:00 | 438.45 | EXIT_EMA400 | -18.45 |
| BUY | 2025-04-29 09:15:00 | 512.15 | 2025-05-06 14:15:00 | 483.20 | EXIT_EMA400 | -28.95 |
| BUY | 2025-05-02 09:15:00 | 499.85 | 2025-05-06 14:15:00 | 483.20 | EXIT_EMA400 | -16.65 |
| BUY | 2025-05-05 12:15:00 | 501.75 | 2025-05-06 14:15:00 | 483.20 | EXIT_EMA400 | -18.55 |
| BUY | 2025-07-29 13:15:00 | 510.80 | 2025-08-04 13:15:00 | 547.39 | TARGET | 36.59 |
| BUY | 2025-07-31 12:15:00 | 513.35 | 2025-08-04 13:15:00 | 555.53 | TARGET | 42.18 |
| BUY | 2025-08-06 11:15:00 | 512.70 | 2025-08-07 09:15:00 | 499.10 | EXIT_EMA400 | -13.60 |
| BUY | 2025-09-15 10:15:00 | 506.00 | 2025-09-16 09:15:00 | 515.63 | TARGET | 9.63 |
| BUY | 2025-09-16 09:15:00 | 517.35 | 2025-09-25 14:15:00 | 506.65 | EXIT_EMA400 | -10.70 |
| SELL | 2026-01-12 09:15:00 | 266.60 | 2026-02-11 09:15:00 | 294.20 | EXIT_EMA400 | -27.60 |
| SELL | 2026-02-10 09:15:00 | 277.60 | 2026-02-11 09:15:00 | 294.20 | EXIT_EMA400 | -16.60 |
