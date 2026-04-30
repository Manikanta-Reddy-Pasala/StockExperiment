# Gallantt Ispat Ltd. (GALLANTT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 858.95
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
- **Total realized P&L (per unit):** 200.75
- **Avg P&L per closed trade:** 40.15

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 09:15:00 | 331.90 | 352.85 | 352.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 09:15:00 | 310.45 | 339.77 | 345.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 334.90 | 326.76 | 335.95 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-12-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 14:15:00 | 378.90 | 342.79 | 342.75 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 10:15:00 | 322.00 | 345.81 | 345.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 11:15:00 | 316.00 | 345.51 | 345.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 14:15:00 | 326.00 | 325.11 | 332.89 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-03 09:15:00 | 316.25 | 325.01 | 332.75 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-05 14:15:00 | 332.45 | 323.42 | 331.21 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-25 15:15:00 | 365.10 | 328.76 | 328.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-26 09:15:00 | 372.85 | 329.20 | 328.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 412.90 | 424.08 | 395.26 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 09:15:00 | 450.00 | 423.91 | 396.16 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 435.00 | 450.17 | 428.67 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-09 10:15:00 | 441.45 | 450.08 | 428.74 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-08-26 12:15:00 | 609.60 | 656.66 | 610.02 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 13:15:00 | 517.00 | 633.70 | 633.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 506.20 | 591.07 | 600.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 09:15:00 | 563.20 | 558.16 | 577.46 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-14 09:15:00 | 546.85 | 560.83 | 574.79 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 14:15:00 | 573.10 | 560.32 | 573.24 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-20 11:15:00 | 560.55 | 560.51 | 573.08 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 14:15:00 | 558.85 | 543.92 | 559.66 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-04 09:15:00 | 567.15 | 544.25 | 559.67 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 10:15:00 | 698.50 | 559.18 | 558.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 12:15:00 | 722.05 | 588.32 | 574.54 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-03 09:15:00 | 316.25 | 2025-02-05 14:15:00 | 332.45 | EXIT_EMA400 | -16.20 |
| BUY | 2025-06-09 10:15:00 | 441.45 | 2025-06-11 09:15:00 | 479.59 | TARGET | 38.14 |
| BUY | 2025-05-12 09:15:00 | 450.00 | 2025-07-21 10:15:00 | 611.52 | TARGET | 161.52 |
| SELL | 2026-01-20 11:15:00 | 560.55 | 2026-01-27 10:15:00 | 522.95 | TARGET | 37.60 |
| SELL | 2026-01-14 09:15:00 | 546.85 | 2026-02-04 09:15:00 | 567.15 | EXIT_EMA400 | -20.30 |
