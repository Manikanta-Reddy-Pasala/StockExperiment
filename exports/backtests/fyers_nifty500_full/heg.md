# H.E.G. Ltd. (HEG.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 594.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 1 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 0 / 3
- **Target hits / EMA400 exits:** 0 / 3
- **Total realized P&L (per unit):** -55.95
- **Avg P&L per closed trade:** -18.65

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 11:15:00 | 501.64 | 425.33 | 425.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 10:15:00 | 509.40 | 460.61 | 446.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 12:15:00 | 466.80 | 470.04 | 453.70 | EMA200 retest candle locked |

### Cycle 2 — SELL (started 2024-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 11:15:00 | 403.90 | 444.63 | 444.73 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 13:15:00 | 559.30 | 441.89 | 441.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 14:15:00 | 576.10 | 443.23 | 442.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-26 13:15:00 | 514.80 | 516.90 | 489.73 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-30 09:15:00 | 519.70 | 516.60 | 490.90 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-01-06 13:15:00 | 493.90 | 517.05 | 495.66 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 12:15:00 | 431.95 | 481.39 | 481.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 09:15:00 | 427.60 | 479.39 | 480.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-06 09:15:00 | 388.05 | 373.37 | 402.78 | EMA200 retest candle locked |

### Cycle 5 — BUY (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 10:15:00 | 477.00 | 414.67 | 414.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 14:15:00 | 483.95 | 417.19 | 415.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 15:15:00 | 458.50 | 458.91 | 444.54 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-05 10:15:00 | 465.25 | 458.99 | 444.72 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-05-06 15:15:00 | 442.25 | 458.88 | 445.51 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 14:15:00 | 461.20 | 505.40 | 505.44 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 11:15:00 | 519.20 | 504.12 | 504.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 11:15:00 | 526.00 | 505.15 | 504.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 506.70 | 510.39 | 507.63 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-29 09:15:00 | 514.65 | 510.36 | 507.66 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 514.65 | 510.36 | 507.66 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-29 15:15:00 | 507.50 | 510.30 | 507.71 | Close below EMA400 |

### Cycle 8 — SELL (started 2026-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 15:15:00 | 524.80 | 549.78 | 549.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 09:15:00 | 523.65 | 549.52 | 549.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 09:15:00 | 547.70 | 547.16 | 548.51 | EMA200 retest candle locked |

### Cycle 9 — BUY (started 2026-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 09:15:00 | 569.00 | 549.66 | 549.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 10:15:00 | 576.25 | 549.92 | 549.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 553.80 | 556.55 | 553.33 | EMA200 retest candle locked |

### Cycle 10 — SELL (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 10:15:00 | 516.60 | 550.41 | 550.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 504.40 | 545.45 | 547.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-27 09:15:00 | 564.00 | 522.48 | 533.95 | EMA200 retest candle locked |

### Cycle 11 — BUY (started 2026-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 11:15:00 | 590.70 | 541.66 | 541.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 607.40 | 544.19 | 542.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 588.55 | 599.54 | 575.50 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-12-30 09:15:00 | 519.70 | 2025-01-06 13:15:00 | 493.90 | EXIT_EMA400 | -25.80 |
| BUY | 2025-05-05 10:15:00 | 465.25 | 2025-05-06 15:15:00 | 442.25 | EXIT_EMA400 | -23.00 |
| BUY | 2025-09-29 09:15:00 | 514.65 | 2025-09-29 15:15:00 | 507.50 | EXIT_EMA400 | -7.15 |
