# PG Electroplast Ltd. (PGEL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 534.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 5 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / EMA400 exits:** 2 / 5
- **Total realized P&L (per unit):** 42.92
- **Avg P&L per closed trade:** 6.13

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 09:15:00 | 183.32 | 160.47 | 160.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-29 09:15:00 | 193.52 | 161.83 | 161.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-22 09:15:00 | 175.57 | 176.28 | 171.06 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-09-25 12:15:00 | 180.39 | 176.35 | 171.35 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 13:15:00 | 174.29 | 176.89 | 172.62 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2023-10-04 14:15:00 | 172.18 | 176.85 | 172.62 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 15:15:00 | 197.52 | 216.46 | 216.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-31 09:15:00 | 194.62 | 216.24 | 216.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 09:15:00 | 174.38 | 173.21 | 184.72 | EMA200 retest candle locked |

### Cycle 3 — BUY (started 2024-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 15:15:00 | 210.46 | 187.90 | 187.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-02 09:15:00 | 218.90 | 188.21 | 188.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 09:15:00 | 559.25 | 594.67 | 530.47 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-09 09:15:00 | 616.10 | 592.37 | 533.58 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-25 10:15:00 | 555.00 | 596.24 | 556.78 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 09:15:00 | 794.30 | 859.26 | 859.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 09:15:00 | 773.20 | 848.86 | 854.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 767.90 | 763.94 | 787.10 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-07 09:15:00 | 736.10 | 787.19 | 791.31 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 11:15:00 | 590.15 | 561.38 | 595.66 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-20 12:15:00 | 586.50 | 561.63 | 595.61 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 566.00 | 557.79 | 579.34 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-17 10:15:00 | 562.55 | 558.55 | 579.19 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 575.15 | 558.71 | 579.17 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-11-17 12:15:00 | 583.15 | 558.96 | 579.19 | Close above EMA400 |

### Cycle 5 — BUY (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 12:15:00 | 628.30 | 578.86 | 578.64 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2026-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-22 10:15:00 | 533.65 | 580.06 | 580.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-22 11:15:00 | 528.85 | 579.55 | 580.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 15:15:00 | 563.55 | 562.84 | 570.47 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-03 11:15:00 | 556.55 | 563.05 | 570.46 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 11:15:00 | 556.55 | 563.05 | 570.46 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-04 09:15:00 | 583.85 | 563.06 | 570.28 | Close above EMA400 |

### Cycle 7 — BUY (started 2026-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 13:15:00 | 624.25 | 575.73 | 575.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 13:15:00 | 632.05 | 598.68 | 589.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 597.30 | 600.15 | 590.53 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-05 14:15:00 | 612.55 | 599.33 | 590.66 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-09 09:15:00 | 579.70 | 600.08 | 591.43 | Close below EMA400 |

### Cycle 8 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 513.05 | 583.62 | 583.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 505.40 | 582.84 | 583.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 10:15:00 | 517.30 | 511.41 | 537.33 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-09-25 12:15:00 | 180.39 | 2023-10-04 14:15:00 | 172.18 | EXIT_EMA400 | -8.21 |
| BUY | 2024-10-09 09:15:00 | 616.10 | 2024-10-25 10:15:00 | 555.00 | EXIT_EMA400 | -61.10 |
| SELL | 2025-08-07 09:15:00 | 736.10 | 2025-08-08 15:15:00 | 570.46 | TARGET | 165.64 |
| SELL | 2025-10-20 12:15:00 | 586.50 | 2025-11-06 10:15:00 | 559.16 | TARGET | 27.34 |
| SELL | 2025-11-17 10:15:00 | 562.55 | 2025-11-17 12:15:00 | 583.15 | EXIT_EMA400 | -20.60 |
| SELL | 2026-02-03 11:15:00 | 556.55 | 2026-02-04 09:15:00 | 583.85 | EXIT_EMA400 | -27.30 |
| BUY | 2026-03-05 14:15:00 | 612.55 | 2026-03-09 09:15:00 | 579.70 | EXIT_EMA400 | -32.85 |
