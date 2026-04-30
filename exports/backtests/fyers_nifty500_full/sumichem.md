# Sumitomo Chemical India Ltd. (SUMICHEM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 422.40
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 5 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 4 / 3
- **Target hits / EMA400 exits:** 4 / 3
- **Total realized P&L (per unit):** 38.55
- **Avg P&L per closed trade:** 5.51

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 09:15:00 | 519.30 | 538.30 | 538.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 09:15:00 | 513.25 | 536.96 | 537.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-30 11:15:00 | 542.30 | 531.74 | 534.69 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-06 10:15:00 | 522.70 | 532.51 | 534.65 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 15:15:00 | 501.00 | 502.82 | 515.34 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-01-28 11:15:00 | 495.10 | 502.70 | 515.10 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-01-30 09:15:00 | 515.90 | 502.62 | 514.33 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 540.00 | 503.86 | 503.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 12:15:00 | 543.20 | 504.97 | 504.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-24 12:15:00 | 537.25 | 537.68 | 525.31 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-16 15:15:00 | 541.05 | 524.53 | 521.89 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 526.30 | 527.53 | 523.98 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-27 09:15:00 | 506.95 | 527.49 | 524.08 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 11:15:00 | 504.05 | 521.14 | 521.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 12:15:00 | 502.60 | 520.96 | 521.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 14:15:00 | 519.05 | 518.49 | 519.75 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-09 10:15:00 | 515.45 | 518.49 | 519.73 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 515.45 | 518.49 | 519.73 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-06-10 10:15:00 | 513.10 | 518.34 | 519.62 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 512.70 | 509.62 | 514.19 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-06-24 09:15:00 | 516.70 | 509.71 | 514.19 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 12:15:00 | 553.90 | 517.03 | 517.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 09:15:00 | 561.55 | 518.52 | 517.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 09:15:00 | 583.55 | 585.63 | 561.70 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-12 15:15:00 | 596.00 | 585.32 | 563.80 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 572.65 | 587.48 | 570.79 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-26 15:15:00 | 565.00 | 587.25 | 570.76 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 11:15:00 | 531.25 | 566.71 | 566.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 13:15:00 | 530.50 | 566.04 | 566.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 13:15:00 | 462.60 | 462.14 | 481.42 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-07 13:15:00 | 459.50 | 464.88 | 478.24 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-08 09:15:00 | 406.60 | 390.92 | 406.57 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 447.15 | 415.90 | 415.76 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-06 10:15:00 | 522.70 | 2025-01-10 09:15:00 | 486.86 | TARGET | 35.84 |
| SELL | 2025-01-28 11:15:00 | 495.10 | 2025-01-30 09:15:00 | 515.90 | EXIT_EMA400 | -20.80 |
| BUY | 2025-05-16 15:15:00 | 541.05 | 2025-05-27 09:15:00 | 506.95 | EXIT_EMA400 | -34.10 |
| SELL | 2025-06-09 10:15:00 | 515.45 | 2025-06-12 10:15:00 | 502.60 | TARGET | 12.85 |
| SELL | 2025-06-10 10:15:00 | 513.10 | 2025-06-13 09:15:00 | 493.55 | TARGET | 19.55 |
| BUY | 2025-08-12 15:15:00 | 596.00 | 2025-08-26 15:15:00 | 565.00 | EXIT_EMA400 | -31.00 |
| SELL | 2026-01-07 13:15:00 | 459.50 | 2026-01-27 14:15:00 | 403.29 | TARGET | 56.21 |
