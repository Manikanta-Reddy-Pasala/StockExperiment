# Granules India Ltd. (GRANULES.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 699.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 5 |
| ENTRY1 | 7 |
| ENTRY2 | 2 |
| EXIT | 7 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 5 / 4
- **Target hits / EMA400 exits:** 4 / 5
- **Total realized P&L (per unit):** 108.15
- **Avg P&L per closed trade:** 12.02

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 12:15:00 | 424.65 | 426.11 | 426.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-06 09:15:00 | 418.55 | 425.97 | 426.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 09:15:00 | 417.90 | 413.70 | 418.82 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-06-04 10:15:00 | 411.00 | 421.33 | 421.85 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 411.00 | 421.33 | 421.85 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-06-04 11:15:00 | 402.80 | 421.15 | 421.75 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-06-05 09:15:00 | 425.25 | 420.95 | 421.63 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 457.40 | 422.49 | 422.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 13:15:00 | 460.60 | 423.53 | 422.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-12 09:15:00 | 657.85 | 663.05 | 613.37 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-10-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 13:15:00 | 553.55 | 587.66 | 587.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 542.50 | 586.57 | 587.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 13:15:00 | 576.45 | 572.74 | 579.22 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-12 14:15:00 | 561.25 | 573.38 | 578.70 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 567.50 | 563.09 | 571.87 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-25 10:15:00 | 572.55 | 563.19 | 571.88 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-12-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 15:15:00 | 595.00 | 575.88 | 575.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-19 10:15:00 | 598.60 | 576.24 | 576.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 12:15:00 | 581.30 | 582.20 | 579.42 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-31 11:15:00 | 587.70 | 582.26 | 579.53 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 587.85 | 592.43 | 586.07 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-13 10:15:00 | 583.90 | 592.35 | 586.06 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 14:15:00 | 518.20 | 582.46 | 582.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 12:15:00 | 510.50 | 567.34 | 573.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 14:15:00 | 507.65 | 507.07 | 529.11 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-25 09:15:00 | 501.30 | 507.19 | 528.21 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-05-14 09:15:00 | 486.90 | 466.34 | 484.97 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 11:15:00 | 529.00 | 496.15 | 496.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 14:15:00 | 531.85 | 497.07 | 496.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 11:15:00 | 511.35 | 513.70 | 506.36 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-16 13:15:00 | 516.45 | 513.37 | 506.51 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-17 10:15:00 | 505.45 | 513.33 | 506.63 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 12:15:00 | 491.75 | 502.10 | 502.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 13:15:00 | 489.15 | 501.97 | 502.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-16 10:15:00 | 496.10 | 493.18 | 496.98 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-22 09:15:00 | 486.85 | 494.34 | 497.13 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 491.55 | 487.91 | 493.05 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-07-31 09:15:00 | 480.60 | 488.02 | 492.93 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 473.05 | 466.30 | 476.05 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-08-28 11:15:00 | 476.15 | 466.50 | 476.05 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 12:15:00 | 518.15 | 483.22 | 483.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 14:15:00 | 520.15 | 483.94 | 483.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-29 11:15:00 | 511.80 | 513.60 | 502.35 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-30 11:15:00 | 523.05 | 513.96 | 502.92 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-11 11:15:00 | 537.20 | 554.63 | 538.54 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-06-04 10:15:00 | 411.00 | 2024-06-05 09:15:00 | 425.25 | EXIT_EMA400 | -14.25 |
| SELL | 2024-06-04 11:15:00 | 402.80 | 2024-06-05 09:15:00 | 425.25 | EXIT_EMA400 | -22.45 |
| SELL | 2024-11-12 14:15:00 | 561.25 | 2024-11-25 10:15:00 | 572.55 | EXIT_EMA400 | -11.30 |
| BUY | 2024-12-31 11:15:00 | 587.70 | 2025-01-06 09:15:00 | 612.22 | TARGET | 24.52 |
| SELL | 2025-03-25 09:15:00 | 501.30 | 2025-05-14 09:15:00 | 486.90 | EXIT_EMA400 | 14.40 |
| BUY | 2025-06-16 13:15:00 | 516.45 | 2025-06-17 10:15:00 | 505.45 | EXIT_EMA400 | -11.00 |
| SELL | 2025-07-22 09:15:00 | 486.85 | 2025-08-01 12:15:00 | 456.00 | TARGET | 30.85 |
| SELL | 2025-07-31 09:15:00 | 480.60 | 2025-08-05 12:15:00 | 443.60 | TARGET | 37.00 |
| BUY | 2025-09-30 11:15:00 | 523.05 | 2025-10-15 09:15:00 | 583.43 | TARGET | 60.38 |
