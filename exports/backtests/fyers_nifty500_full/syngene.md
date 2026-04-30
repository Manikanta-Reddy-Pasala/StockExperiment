# Syngene International Ltd. (SYNGENE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 469.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 3 |
| ENTRY1 | 2 |
| ENTRY2 | 1 |
| EXIT | 2 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 1 / 2
- **Target hits / EMA400 exits:** 1 / 2
- **Total realized P&L (per unit):** -0.16
- **Avg P&L per closed trade:** -0.05

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 11:15:00 | 855.50 | 873.01 | 873.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 12:15:00 | 849.60 | 872.78 | 872.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-01 12:15:00 | 872.95 | 871.33 | 872.17 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-02 09:15:00 | 862.35 | 871.39 | 872.19 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 862.35 | 871.39 | 872.19 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-02 14:15:00 | 873.45 | 871.21 | 872.08 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 13:15:00 | 709.50 | 660.11 | 659.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 14:15:00 | 711.45 | 660.62 | 660.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 10:15:00 | 674.25 | 674.81 | 668.14 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 11:15:00 | 632.50 | 664.96 | 664.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 12:15:00 | 629.70 | 664.60 | 664.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-04 09:15:00 | 663.35 | 658.91 | 661.71 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-04 15:15:00 | 653.80 | 658.85 | 661.60 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 656.75 | 658.83 | 661.57 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-05 11:15:00 | 647.55 | 658.67 | 661.46 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 658.70 | 656.53 | 659.98 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-11 10:15:00 | 660.00 | 656.57 | 659.98 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 15:15:00 | 657.00 | 645.96 | 645.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 661.65 | 648.28 | 647.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 10:15:00 | 646.60 | 650.26 | 648.42 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2026-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 10:15:00 | 625.45 | 646.65 | 646.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 11:15:00 | 618.45 | 641.88 | 644.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 14:15:00 | 413.80 | 413.60 | 450.37 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-02 09:15:00 | 862.35 | 2025-01-02 14:15:00 | 873.45 | EXIT_EMA400 | -11.10 |
| SELL | 2025-09-04 15:15:00 | 653.80 | 2025-09-09 11:15:00 | 630.41 | TARGET | 23.39 |
| SELL | 2025-09-05 11:15:00 | 647.55 | 2025-09-11 10:15:00 | 660.00 | EXIT_EMA400 | -12.45 |
