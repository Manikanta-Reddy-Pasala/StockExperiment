# Choice International Ltd. (CHOICEIN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 664.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 2 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / EMA400 exits:** 2 / 3
- **Total realized P&L (per unit):** 114.59
- **Avg P&L per closed trade:** 22.92

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-12 13:15:00 | 492.55 | 518.78 | 518.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 09:15:00 | 491.85 | 516.90 | 517.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 11:15:00 | 490.60 | 490.34 | 499.39 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-07 09:15:00 | 482.80 | 497.38 | 500.43 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 497.90 | 496.45 | 499.85 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-08 12:15:00 | 501.55 | 496.53 | 499.84 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 15:15:00 | 538.00 | 502.82 | 502.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 11:15:00 | 542.50 | 503.93 | 503.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 15:15:00 | 691.10 | 691.27 | 658.75 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-03 12:15:00 | 693.85 | 691.31 | 659.41 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 776.35 | 800.31 | 775.56 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-30 13:15:00 | 769.05 | 799.55 | 775.55 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 09:15:00 | 726.70 | 790.97 | 791.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 15:15:00 | 721.80 | 783.51 | 787.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 10:15:00 | 775.15 | 773.80 | 781.25 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-18 15:15:00 | 770.00 | 773.78 | 781.06 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-19 10:15:00 | 782.30 | 773.91 | 781.06 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 10:15:00 | 838.60 | 786.72 | 786.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 11:15:00 | 840.00 | 787.25 | 786.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-12 10:15:00 | 808.20 | 809.99 | 800.24 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-14 09:15:00 | 818.20 | 810.23 | 800.98 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-20 14:15:00 | 799.45 | 812.56 | 803.39 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 13:15:00 | 772.60 | 796.27 | 796.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 15:15:00 | 764.00 | 795.64 | 795.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 12:15:00 | 793.40 | 778.58 | 785.44 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-23 14:15:00 | 766.20 | 782.05 | 786.56 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-10 09:15:00 | 703.75 | 667.08 | 702.06 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-04-07 09:15:00 | 482.80 | 2025-04-08 12:15:00 | 501.55 | EXIT_EMA400 | -18.75 |
| BUY | 2025-07-03 12:15:00 | 693.85 | 2025-08-12 12:15:00 | 797.17 | TARGET | 103.32 |
| SELL | 2025-12-18 15:15:00 | 770.00 | 2025-12-19 10:15:00 | 782.30 | EXIT_EMA400 | -12.30 |
| BUY | 2026-01-14 09:15:00 | 818.20 | 2026-01-20 14:15:00 | 799.45 | EXIT_EMA400 | -18.75 |
| SELL | 2026-02-23 14:15:00 | 766.20 | 2026-03-04 09:15:00 | 705.13 | TARGET | 61.07 |
