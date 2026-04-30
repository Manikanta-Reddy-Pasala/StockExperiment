# Gallantt Ispat Ltd. (GALLANTT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5006 bars)
- **Last close:** 861.85
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT3 | 3 |
| ENTRY1 | 3 |
| ENTRY2 | 2 |
| EXIT | 3 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 3 / 2
- **Target hits / EMA400 exits:** 3 / 2
- **Total realized P&L (per unit):** 200.16
- **Avg P&L per closed trade:** 40.03

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 14:15:00 | 320.30 | 351.36 | 351.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 10:15:00 | 319.40 | 350.49 | 350.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 334.90 | 326.81 | 335.62 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-12-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 13:15:00 | 379.00 | 342.46 | 342.31 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 11:15:00 | 316.00 | 345.58 | 345.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 13:15:00 | 315.55 | 345.02 | 345.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 15:15:00 | 327.20 | 325.29 | 333.13 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-03 09:15:00 | 316.25 | 325.20 | 333.05 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-05 14:15:00 | 332.45 | 323.58 | 331.47 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-25 15:15:00 | 365.05 | 328.83 | 328.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-26 09:15:00 | 372.85 | 329.26 | 328.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 412.90 | 424.14 | 395.34 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 09:15:00 | 450.00 | 423.95 | 396.23 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 435.00 | 450.29 | 428.77 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-09 10:15:00 | 441.45 | 450.20 | 428.83 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-08-26 12:15:00 | 609.60 | 656.59 | 609.98 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 13:15:00 | 517.00 | 633.75 | 633.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 506.20 | 590.97 | 600.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 09:15:00 | 563.20 | 558.09 | 577.41 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-14 09:15:00 | 546.85 | 560.74 | 574.72 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 14:15:00 | 573.10 | 560.29 | 573.21 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-20 11:15:00 | 560.55 | 560.48 | 573.05 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 14:15:00 | 558.85 | 545.70 | 561.10 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-04 09:15:00 | 567.15 | 546.04 | 561.11 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 10:15:00 | 698.50 | 559.27 | 559.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 12:15:00 | 722.05 | 588.41 | 574.84 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-03 09:15:00 | 316.25 | 2025-02-05 14:15:00 | 332.45 | EXIT_EMA400 | -16.20 |
| BUY | 2025-06-09 10:15:00 | 441.45 | 2025-06-11 09:15:00 | 479.31 | TARGET | 37.86 |
| BUY | 2025-05-12 09:15:00 | 450.00 | 2025-07-21 10:15:00 | 611.31 | TARGET | 161.31 |
| SELL | 2026-01-20 11:15:00 | 560.55 | 2026-01-27 10:15:00 | 523.05 | TARGET | 37.50 |
| SELL | 2026-01-14 09:15:00 | 546.85 | 2026-02-04 09:15:00 | 567.15 | EXIT_EMA400 | -20.30 |
