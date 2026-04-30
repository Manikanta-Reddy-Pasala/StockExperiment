# Syrma SGS Technology Ltd. (SYRMA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 958.95
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT3 | 1 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / EMA400 exits:** 2 / 3
- **Total realized P&L (per unit):** 92.75
- **Avg P&L per closed trade:** 18.55

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-07 13:15:00 | 527.50 | 602.33 | 602.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 11:15:00 | 517.90 | 598.40 | 600.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-30 09:15:00 | 487.70 | 486.52 | 506.75 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-07 12:15:00 | 478.25 | 487.07 | 504.15 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-05-24 13:15:00 | 482.95 | 454.71 | 478.91 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 11:15:00 | 507.40 | 481.25 | 481.18 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-07-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 14:15:00 | 453.60 | 481.89 | 482.00 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-01 15:15:00 | 501.40 | 481.55 | 481.46 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 09:15:00 | 456.90 | 481.23 | 481.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 14:15:00 | 426.60 | 479.42 | 480.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-23 11:15:00 | 456.75 | 447.41 | 460.44 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-26 10:15:00 | 440.00 | 447.61 | 460.16 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-09-12 13:15:00 | 455.55 | 437.76 | 449.67 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-11-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 11:15:00 | 507.35 | 439.63 | 439.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-05 13:15:00 | 509.00 | 440.98 | 440.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 09:15:00 | 564.35 | 588.86 | 556.87 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 09:15:00 | 412.00 | 539.21 | 539.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 10:15:00 | 405.30 | 537.88 | 538.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-30 09:15:00 | 531.80 | 530.39 | 534.85 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-30 11:15:00 | 508.75 | 530.04 | 534.63 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 523.55 | 529.71 | 534.40 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-02-03 09:15:00 | 534.95 | 529.66 | 534.16 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 538.00 | 478.68 | 478.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 545.05 | 481.14 | 479.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 518.90 | 526.52 | 511.89 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-13 10:15:00 | 523.65 | 526.50 | 511.95 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-19 12:15:00 | 512.80 | 525.64 | 513.51 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 12:15:00 | 734.95 | 795.32 | 795.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 727.30 | 792.84 | 794.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 11:15:00 | 753.45 | 753.07 | 769.12 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-08 09:15:00 | 744.85 | 752.55 | 767.40 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-30 09:15:00 | 769.95 | 713.32 | 737.70 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-02-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 13:15:00 | 878.85 | 756.39 | 756.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 883.65 | 759.93 | 757.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 795.90 | 815.52 | 794.20 | EMA200 retest candle locked |

### Cycle 11 — SELL (started 2026-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 09:15:00 | 743.25 | 779.54 | 779.67 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 09:15:00 | 816.00 | 779.58 | 779.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 837.55 | 786.97 | 783.51 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-05-07 12:15:00 | 478.25 | 2024-05-13 11:15:00 | 400.55 | TARGET | 77.70 |
| SELL | 2024-08-26 10:15:00 | 440.00 | 2024-09-12 13:15:00 | 455.55 | EXIT_EMA400 | -15.55 |
| SELL | 2025-01-30 11:15:00 | 508.75 | 2025-02-03 09:15:00 | 534.95 | EXIT_EMA400 | -26.20 |
| BUY | 2025-06-13 10:15:00 | 523.65 | 2025-06-19 12:15:00 | 512.80 | EXIT_EMA400 | -10.85 |
| SELL | 2026-01-08 09:15:00 | 744.85 | 2026-01-20 09:15:00 | 677.20 | TARGET | 67.65 |
