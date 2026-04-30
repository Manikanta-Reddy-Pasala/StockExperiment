# LIC Housing Finance Ltd. (LICHSGFIN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 554.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 2 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 1 / 2
- **Target hits / EMA400 exits:** 1 / 2
- **Total realized P&L (per unit):** -5.11
- **Avg P&L per closed trade:** -1.70

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 09:15:00 | 658.20 | 711.48 | 711.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 10:15:00 | 653.80 | 691.70 | 697.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-24 11:15:00 | 688.10 | 687.68 | 694.52 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-24 14:15:00 | 685.15 | 687.68 | 694.41 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-10-30 09:15:00 | 654.40 | 631.72 | 653.22 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 10:15:00 | 615.40 | 562.28 | 562.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 09:15:00 | 620.00 | 577.50 | 570.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 15:15:00 | 584.00 | 584.86 | 575.66 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-07 10:15:00 | 592.75 | 584.94 | 575.80 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 578.00 | 585.71 | 576.68 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-08 15:15:00 | 574.50 | 585.60 | 576.67 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 10:15:00 | 574.40 | 600.84 | 600.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 11:15:00 | 573.50 | 600.56 | 600.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 573.30 | 569.89 | 579.67 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-26 13:15:00 | 564.30 | 574.63 | 579.65 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 577.20 | 573.44 | 578.47 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-03 09:15:00 | 578.95 | 573.50 | 578.47 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-04-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 15:15:00 | 564.80 | 520.52 | 520.33 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-24 14:15:00 | 685.15 | 2024-09-26 13:15:00 | 657.36 | TARGET | 27.79 |
| BUY | 2025-05-07 10:15:00 | 592.75 | 2025-05-08 15:15:00 | 574.50 | EXIT_EMA400 | -18.25 |
| SELL | 2025-09-26 13:15:00 | 564.30 | 2025-10-03 09:15:00 | 578.95 | EXIT_EMA400 | -14.65 |
