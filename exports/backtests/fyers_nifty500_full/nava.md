# Nava Ltd. (NAVA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 660.20
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** -62.25
- **Avg P&L per closed trade:** -10.38

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 13:15:00 | 469.30 | 485.86 | 485.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 15:15:00 | 467.28 | 485.54 | 485.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 487.00 | 482.86 | 484.28 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-26 14:15:00 | 511.48 | 485.65 | 485.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 15:15:00 | 515.00 | 485.94 | 485.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 09:15:00 | 498.00 | 504.47 | 497.29 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-16 09:15:00 | 512.45 | 504.36 | 497.48 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 507.00 | 504.89 | 498.44 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-12-20 15:15:00 | 496.00 | 504.79 | 498.80 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 12:15:00 | 465.75 | 495.45 | 495.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 12:15:00 | 465.15 | 493.57 | 494.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-30 09:15:00 | 452.80 | 452.11 | 468.67 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-03 10:15:00 | 422.45 | 448.91 | 465.26 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-06 14:15:00 | 439.45 | 412.08 | 431.17 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 517.95 | 439.61 | 439.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 09:15:00 | 534.65 | 450.48 | 444.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 453.10 | 461.86 | 451.51 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-08 12:15:00 | 475.15 | 461.62 | 451.89 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-04-30 12:15:00 | 459.00 | 471.26 | 461.40 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 14:15:00 | 617.45 | 636.93 | 636.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 09:15:00 | 609.30 | 636.45 | 636.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 13:15:00 | 559.15 | 550.76 | 577.77 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-20 13:15:00 | 542.75 | 567.69 | 573.13 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-30 09:15:00 | 567.50 | 559.35 | 567.34 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-02 12:15:00 | 576.05 | 568.01 | 568.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-02 13:15:00 | 578.00 | 568.11 | 568.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 10:15:00 | 558.55 | 568.21 | 568.11 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2026-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 12:15:00 | 557.55 | 567.98 | 568.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 541.50 | 567.69 | 567.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 15:15:00 | 569.00 | 565.88 | 566.89 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-13 09:15:00 | 560.00 | 567.07 | 567.43 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 560.00 | 567.07 | 567.43 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-03-13 10:15:00 | 556.30 | 566.96 | 567.38 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 561.50 | 563.49 | 565.51 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-03-18 13:15:00 | 566.50 | 563.53 | 565.49 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 13:15:00 | 604.20 | 565.01 | 564.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 10:15:00 | 614.50 | 566.59 | 565.75 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-12-16 09:15:00 | 512.45 | 2024-12-20 15:15:00 | 496.00 | EXIT_EMA400 | -16.45 |
| SELL | 2025-02-03 10:15:00 | 422.45 | 2025-03-06 14:15:00 | 439.45 | EXIT_EMA400 | -17.00 |
| BUY | 2025-04-08 12:15:00 | 475.15 | 2025-04-30 12:15:00 | 459.00 | EXIT_EMA400 | -16.15 |
| SELL | 2026-01-20 13:15:00 | 542.75 | 2026-01-30 09:15:00 | 567.50 | EXIT_EMA400 | -24.75 |
| SELL | 2026-03-13 09:15:00 | 560.00 | 2026-03-16 13:15:00 | 537.70 | TARGET | 22.30 |
| SELL | 2026-03-13 10:15:00 | 556.30 | 2026-03-18 13:15:00 | 566.50 | EXIT_EMA400 | -10.20 |
