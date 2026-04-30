# Granules India Ltd. (GRANULES.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 696.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 4 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 6 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 5 / 2
- **Target hits / EMA400 exits:** 4 / 3
- **Total realized P&L (per unit):** 145.36
- **Avg P&L per closed trade:** 20.77

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 15:15:00 | 556.00 | 586.96 | 587.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 542.90 | 586.52 | 586.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 13:15:00 | 576.45 | 572.65 | 578.77 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-12 14:15:00 | 561.25 | 573.32 | 578.31 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 567.50 | 563.07 | 571.58 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-25 10:15:00 | 572.50 | 563.17 | 571.58 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 13:15:00 | 592.50 | 575.48 | 575.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-18 14:15:00 | 596.70 | 575.69 | 575.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 12:15:00 | 581.30 | 582.19 | 579.29 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-31 11:15:00 | 587.70 | 582.22 | 579.39 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 11:15:00 | 589.85 | 592.42 | 585.81 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-13 10:15:00 | 583.90 | 592.32 | 585.95 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 14:15:00 | 518.80 | 582.42 | 582.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 12:15:00 | 510.50 | 566.74 | 573.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 14:15:00 | 507.65 | 506.92 | 528.85 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-25 09:15:00 | 501.30 | 507.06 | 527.96 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-05-14 09:15:00 | 486.75 | 466.30 | 484.87 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 11:15:00 | 529.55 | 496.10 | 496.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 14:15:00 | 531.85 | 497.03 | 496.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 11:15:00 | 511.35 | 513.68 | 506.30 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-16 13:15:00 | 516.45 | 513.34 | 506.45 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-17 10:15:00 | 505.45 | 513.30 | 506.57 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 13:15:00 | 489.15 | 501.95 | 502.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 09:15:00 | 486.05 | 499.97 | 500.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-16 10:15:00 | 496.10 | 493.15 | 496.94 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-22 09:15:00 | 486.85 | 494.35 | 497.12 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 491.55 | 487.89 | 493.02 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-07-31 09:15:00 | 480.60 | 488.00 | 492.91 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 473.05 | 466.29 | 476.03 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-08-28 11:15:00 | 476.15 | 466.48 | 476.03 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 12:15:00 | 518.15 | 483.19 | 483.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 14:15:00 | 520.20 | 483.91 | 483.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-29 11:15:00 | 511.75 | 513.60 | 502.34 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-30 11:15:00 | 523.05 | 513.96 | 502.91 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-11 11:15:00 | 537.20 | 554.63 | 538.55 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-12 14:15:00 | 561.25 | 2024-11-25 10:15:00 | 572.50 | EXIT_EMA400 | -11.25 |
| BUY | 2024-12-31 11:15:00 | 587.70 | 2025-01-03 09:15:00 | 612.63 | TARGET | 24.93 |
| SELL | 2025-03-25 09:15:00 | 501.30 | 2025-05-14 09:15:00 | 486.75 | EXIT_EMA400 | 14.55 |
| BUY | 2025-06-16 13:15:00 | 516.45 | 2025-06-17 10:15:00 | 505.45 | EXIT_EMA400 | -11.00 |
| SELL | 2025-07-22 09:15:00 | 486.85 | 2025-08-01 11:15:00 | 456.05 | TARGET | 30.80 |
| SELL | 2025-07-31 09:15:00 | 480.60 | 2025-08-05 12:15:00 | 443.68 | TARGET | 36.92 |
| BUY | 2025-09-30 11:15:00 | 523.05 | 2025-10-15 09:15:00 | 583.46 | TARGET | 60.41 |
