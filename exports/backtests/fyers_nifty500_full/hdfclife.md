# HDFC Life Insurance Company Ltd. (HDFCLIFE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 589.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 5 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** -20.50
- **Avg P&L per closed trade:** -3.42

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 10:15:00 | 687.50 | 704.19 | 704.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-26 14:15:00 | 682.90 | 703.50 | 703.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-16 09:15:00 | 653.10 | 625.08 | 646.08 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-21 10:15:00 | 626.55 | 627.74 | 645.31 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 636.55 | 624.55 | 639.23 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-02-01 09:15:00 | 643.00 | 625.50 | 639.21 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 09:15:00 | 683.80 | 633.64 | 633.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-01 09:15:00 | 692.40 | 647.01 | 640.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 12:15:00 | 656.90 | 658.17 | 647.74 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-07 13:15:00 | 663.05 | 658.22 | 647.82 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 760.30 | 776.50 | 756.56 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-14 13:15:00 | 763.25 | 775.90 | 756.65 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 757.25 | 775.35 | 756.85 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-15 12:15:00 | 752.85 | 775.12 | 756.83 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 11:15:00 | 750.80 | 766.27 | 766.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 742.70 | 763.41 | 764.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 15:15:00 | 763.35 | 761.98 | 764.01 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-16 09:15:00 | 741.20 | 761.77 | 763.90 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 756.55 | 754.21 | 759.17 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-29 12:15:00 | 760.05 | 754.31 | 759.15 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 785.45 | 759.54 | 759.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 13:15:00 | 786.70 | 759.81 | 759.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 11:15:00 | 760.75 | 761.97 | 760.84 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-01 14:15:00 | 767.55 | 762.01 | 760.87 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 767.55 | 762.01 | 760.87 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-12-02 09:15:00 | 757.30 | 762.00 | 760.87 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 14:15:00 | 747.05 | 761.02 | 761.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 743.40 | 760.71 | 760.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 09:15:00 | 762.00 | 758.12 | 759.47 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-09 12:15:00 | 750.80 | 760.00 | 760.31 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-12 09:15:00 | 761.05 | 759.71 | 760.16 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-21 10:15:00 | 626.55 | 2025-02-01 09:15:00 | 643.00 | EXIT_EMA400 | -16.45 |
| BUY | 2025-04-07 13:15:00 | 663.05 | 2025-04-16 09:15:00 | 708.75 | TARGET | 45.70 |
| BUY | 2025-07-14 13:15:00 | 763.25 | 2025-07-15 12:15:00 | 752.85 | EXIT_EMA400 | -10.40 |
| SELL | 2025-10-16 09:15:00 | 741.20 | 2025-10-29 12:15:00 | 760.05 | EXIT_EMA400 | -18.85 |
| BUY | 2025-12-01 14:15:00 | 767.55 | 2025-12-02 09:15:00 | 757.30 | EXIT_EMA400 | -10.25 |
| SELL | 2026-01-09 12:15:00 | 750.80 | 2026-01-12 09:15:00 | 761.05 | EXIT_EMA400 | -10.25 |
