# Sona BLW Precision Forgings Ltd. (SONACOMS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 605.35
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 6 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 6 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 0 / 7
- **Target hits / EMA400 exits:** 0 / 7
- **Total realized P&L (per unit):** -133.65
- **Avg P&L per closed trade:** -19.09

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 12:15:00 | 637.65 | 691.22 | 691.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 09:15:00 | 631.95 | 689.10 | 690.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 09:15:00 | 708.30 | 680.23 | 685.51 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-11-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-08 14:15:00 | 705.05 | 689.08 | 689.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-11 11:15:00 | 707.35 | 689.67 | 689.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-12 12:15:00 | 689.15 | 690.49 | 689.78 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 14:15:00 | 666.00 | 688.93 | 689.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 10:15:00 | 660.95 | 686.84 | 687.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 09:15:00 | 693.90 | 685.92 | 687.44 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-28 12:15:00 | 667.15 | 684.00 | 686.13 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 684.20 | 681.36 | 684.46 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-04 10:15:00 | 692.90 | 681.47 | 684.51 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 11:15:00 | 527.25 | 501.11 | 501.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 09:15:00 | 531.70 | 502.43 | 501.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 09:15:00 | 512.30 | 521.20 | 513.03 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-11 09:15:00 | 532.80 | 520.51 | 513.47 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-12 12:15:00 | 513.00 | 520.47 | 513.79 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 09:15:00 | 480.30 | 508.47 | 508.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 11:15:00 | 474.20 | 498.30 | 502.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-17 11:15:00 | 488.45 | 477.59 | 489.16 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-23 09:15:00 | 470.30 | 478.49 | 488.24 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 12:15:00 | 487.70 | 478.45 | 488.07 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-07-23 13:15:00 | 489.10 | 478.56 | 488.08 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 09:15:00 | 481.85 | 449.24 | 449.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 12:15:00 | 487.35 | 461.00 | 455.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 11:15:00 | 489.30 | 490.69 | 477.77 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-08 09:15:00 | 496.25 | 490.80 | 478.14 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 480.50 | 490.55 | 478.45 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-12-09 11:15:00 | 487.00 | 490.45 | 478.52 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 479.15 | 490.22 | 478.59 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-12-10 10:15:00 | 476.45 | 489.88 | 478.58 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 10:15:00 | 460.65 | 477.74 | 477.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 10:15:00 | 452.40 | 475.45 | 476.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 15:15:00 | 470.00 | 469.69 | 473.38 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-23 09:15:00 | 463.95 | 469.63 | 473.33 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 470.80 | 469.06 | 472.91 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-01-27 10:15:00 | 477.45 | 469.14 | 472.94 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 500.00 | 476.28 | 476.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 528.40 | 477.03 | 476.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 11:15:00 | 513.15 | 517.45 | 503.33 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-11 09:15:00 | 527.10 | 514.57 | 504.29 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 506.05 | 514.76 | 504.74 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-13 09:15:00 | 501.65 | 514.38 | 504.90 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-28 12:15:00 | 667.15 | 2024-12-04 10:15:00 | 692.90 | EXIT_EMA400 | -25.75 |
| BUY | 2025-06-11 09:15:00 | 532.80 | 2025-06-12 12:15:00 | 513.00 | EXIT_EMA400 | -19.80 |
| SELL | 2025-07-23 09:15:00 | 470.30 | 2025-07-23 13:15:00 | 489.10 | EXIT_EMA400 | -18.80 |
| BUY | 2025-12-08 09:15:00 | 496.25 | 2025-12-10 10:15:00 | 476.45 | EXIT_EMA400 | -19.80 |
| BUY | 2025-12-09 11:15:00 | 487.00 | 2025-12-10 10:15:00 | 476.45 | EXIT_EMA400 | -10.55 |
| SELL | 2026-01-23 09:15:00 | 463.95 | 2026-01-27 10:15:00 | 477.45 | EXIT_EMA400 | -13.50 |
| BUY | 2026-03-11 09:15:00 | 527.10 | 2026-03-13 09:15:00 | 501.65 | EXIT_EMA400 | -25.45 |
