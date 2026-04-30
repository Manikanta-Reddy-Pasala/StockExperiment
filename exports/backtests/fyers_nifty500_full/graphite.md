# Graphite India Ltd. (GRAPHITE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 703.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 2 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 6 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / EMA400 exits:** 2 / 5
- **Total realized P&L (per unit):** 19.05
- **Avg P&L per closed trade:** 2.72

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 14:15:00 | 597.60 | 540.14 | 540.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-25 09:15:00 | 618.60 | 541.48 | 540.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 09:15:00 | 552.75 | 559.95 | 551.43 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-09 09:15:00 | 579.90 | 558.57 | 551.28 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-21 12:15:00 | 556.30 | 567.99 | 558.60 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 09:15:00 | 505.55 | 551.14 | 551.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 10:15:00 | 501.55 | 550.64 | 550.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-07 09:15:00 | 541.20 | 540.81 | 545.35 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-07 14:15:00 | 534.75 | 540.68 | 545.17 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 15:15:00 | 521.25 | 509.25 | 524.11 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-02 10:15:00 | 518.45 | 509.45 | 524.06 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-12-02 15:15:00 | 526.95 | 509.94 | 523.95 | Close above EMA400 |

### Cycle 3 — BUY (started 2024-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-12 14:15:00 | 559.30 | 535.08 | 534.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 10:15:00 | 563.50 | 535.78 | 535.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 10:15:00 | 550.65 | 554.11 | 546.89 | EMA200 retest candle locked |

### Cycle 4 — SELL (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-16 10:15:00 | 515.80 | 542.31 | 542.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-16 11:15:00 | 512.40 | 542.01 | 542.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-06 09:15:00 | 513.50 | 504.59 | 518.77 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-07 09:15:00 | 500.15 | 504.68 | 518.33 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-18 14:15:00 | 460.75 | 432.56 | 458.96 | Close above EMA400 |

### Cycle 5 — BUY (started 2025-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 10:15:00 | 556.20 | 468.36 | 467.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 11:15:00 | 566.45 | 469.34 | 468.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 525.00 | 527.16 | 506.75 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-16 12:15:00 | 532.20 | 526.89 | 507.61 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 545.25 | 563.12 | 545.02 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-25 12:15:00 | 544.40 | 562.93 | 545.01 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 13:15:00 | 510.60 | 539.93 | 540.07 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 13:15:00 | 576.00 | 537.64 | 537.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 09:15:00 | 578.20 | 545.20 | 541.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-29 11:15:00 | 545.25 | 547.58 | 543.17 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-30 14:15:00 | 555.30 | 547.61 | 543.40 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-10-03 11:15:00 | 543.45 | 547.84 | 543.74 | Close below EMA400 |

### Cycle 8 — SELL (started 2025-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 14:15:00 | 537.65 | 561.50 | 561.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 15:15:00 | 536.20 | 561.25 | 561.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 14:15:00 | 550.15 | 549.67 | 554.61 | EMA200 retest candle locked |

### Cycle 9 — BUY (started 2025-12-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 15:15:00 | 598.00 | 558.62 | 558.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 09:15:00 | 609.55 | 561.25 | 559.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 12:15:00 | 600.15 | 601.83 | 584.20 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-14 09:15:00 | 632.00 | 601.94 | 584.60 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-02-01 09:15:00 | 597.00 | 619.94 | 600.84 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-10-09 09:15:00 | 579.90 | 2024-10-21 12:15:00 | 556.30 | EXIT_EMA400 | -23.60 |
| SELL | 2024-11-07 14:15:00 | 534.75 | 2024-11-12 13:15:00 | 503.49 | TARGET | 31.26 |
| SELL | 2024-12-02 10:15:00 | 518.45 | 2024-12-02 15:15:00 | 526.95 | EXIT_EMA400 | -8.50 |
| SELL | 2025-02-07 09:15:00 | 500.15 | 2025-02-12 09:15:00 | 445.61 | TARGET | 54.54 |
| BUY | 2025-06-16 12:15:00 | 532.20 | 2025-07-25 12:15:00 | 544.40 | EXIT_EMA400 | 12.20 |
| BUY | 2025-09-30 14:15:00 | 555.30 | 2025-10-03 11:15:00 | 543.45 | EXIT_EMA400 | -11.85 |
| BUY | 2026-01-14 09:15:00 | 632.00 | 2026-02-01 09:15:00 | 597.00 | EXIT_EMA400 | -35.00 |
