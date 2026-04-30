# Graphite India Ltd. (GRAPHITE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 707.10
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 2 |
| ENTRY1 | 8 |
| ENTRY2 | 1 |
| EXIT | 8 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 5 / 4
- **Target hits / EMA400 exits:** 4 / 5
- **Total realized P&L (per unit):** 80.46
- **Avg P&L per closed trade:** 8.94

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 14:15:00 | 568.85 | 613.62 | 613.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 15:15:00 | 567.00 | 613.15 | 613.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-19 12:15:00 | 589.85 | 585.18 | 595.61 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-06-26 15:15:00 | 578.05 | 585.86 | 594.22 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 552.25 | 537.14 | 555.13 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-08-13 11:15:00 | 549.05 | 537.26 | 555.10 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-09-16 09:15:00 | 537.95 | 520.87 | 535.27 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 15:15:00 | 599.50 | 545.38 | 545.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 12:15:00 | 604.25 | 555.60 | 550.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 09:15:00 | 552.75 | 559.99 | 553.50 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-09 09:15:00 | 579.95 | 558.61 | 553.22 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-21 11:15:00 | 559.10 | 568.17 | 560.08 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 11:15:00 | 504.25 | 553.55 | 553.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-28 15:15:00 | 502.55 | 551.62 | 552.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-07 09:15:00 | 541.20 | 541.07 | 546.48 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-07 11:15:00 | 537.70 | 541.04 | 546.41 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-02 15:15:00 | 526.95 | 510.04 | 524.61 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 10:15:00 | 563.50 | 535.84 | 535.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 15:15:00 | 565.25 | 537.20 | 536.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 10:15:00 | 550.65 | 554.15 | 547.20 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-07 12:15:00 | 557.60 | 554.26 | 547.57 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-01-08 09:15:00 | 547.00 | 554.21 | 547.68 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-16 09:15:00 | 515.10 | 542.57 | 542.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-16 12:15:00 | 512.00 | 541.71 | 542.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-06 09:15:00 | 513.50 | 506.29 | 520.27 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-06 13:15:00 | 501.30 | 506.42 | 520.06 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-18 14:15:00 | 460.70 | 432.90 | 459.62 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 10:15:00 | 556.20 | 468.41 | 468.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 11:15:00 | 566.10 | 469.38 | 468.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 525.00 | 527.13 | 506.82 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-16 11:15:00 | 530.50 | 526.81 | 507.55 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 545.25 | 563.10 | 545.04 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-25 12:15:00 | 544.40 | 562.92 | 545.04 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 12:15:00 | 510.50 | 540.24 | 540.24 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 14:15:00 | 569.75 | 537.97 | 537.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 09:15:00 | 578.20 | 545.22 | 541.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-29 11:15:00 | 545.25 | 547.59 | 543.19 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-30 14:15:00 | 555.30 | 547.63 | 543.42 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-10-03 11:15:00 | 543.45 | 547.82 | 543.75 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 14:15:00 | 537.65 | 561.49 | 561.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 15:15:00 | 536.20 | 561.24 | 561.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 14:15:00 | 550.10 | 549.68 | 554.63 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2025-12-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 15:15:00 | 601.75 | 558.66 | 558.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 09:15:00 | 609.55 | 561.29 | 559.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 12:15:00 | 600.15 | 601.85 | 584.22 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-14 09:15:00 | 632.00 | 602.00 | 584.64 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-02-02 09:15:00 | 592.80 | 620.00 | 600.88 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-06-26 15:15:00 | 578.05 | 2024-07-19 15:15:00 | 529.55 | TARGET | 48.50 |
| SELL | 2024-08-13 11:15:00 | 549.05 | 2024-08-13 15:15:00 | 530.91 | TARGET | 18.14 |
| BUY | 2024-10-09 09:15:00 | 579.95 | 2024-10-21 11:15:00 | 559.10 | EXIT_EMA400 | -20.85 |
| SELL | 2024-11-07 11:15:00 | 537.70 | 2024-11-11 13:15:00 | 511.57 | TARGET | 26.13 |
| BUY | 2025-01-07 12:15:00 | 557.60 | 2025-01-08 09:15:00 | 547.00 | EXIT_EMA400 | -10.60 |
| SELL | 2025-02-06 13:15:00 | 501.30 | 2025-02-12 09:15:00 | 445.02 | TARGET | 56.28 |
| BUY | 2025-06-16 11:15:00 | 530.50 | 2025-07-25 12:15:00 | 544.40 | EXIT_EMA400 | 13.90 |
| BUY | 2025-09-30 14:15:00 | 555.30 | 2025-10-03 11:15:00 | 543.45 | EXIT_EMA400 | -11.85 |
| BUY | 2026-01-14 09:15:00 | 632.00 | 2026-02-02 09:15:00 | 592.80 | EXIT_EMA400 | -39.20 |
