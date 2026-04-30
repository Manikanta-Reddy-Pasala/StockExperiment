# Choice International Ltd. (CHOICEIN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 663.45
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 2 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 3 / 2
- **Target hits / EMA400 exits:** 3 / 2
- **Total realized P&L (per unit):** 156.22
- **Avg P&L per closed trade:** 31.24

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-04 09:15:00 | 191.23 | 180.97 | 180.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-04 10:15:00 | 192.50 | 181.08 | 181.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-23 10:15:00 | 207.62 | 207.79 | 200.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-11-08 09:15:00 | 210.55 | 205.70 | 201.47 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-03-11 14:15:00 | 261.65 | 272.75 | 262.16 | Close below EMA400 |

### Cycle 2 — SELL (started 2025-02-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-12 15:15:00 | 495.80 | 518.86 | 518.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 09:15:00 | 491.85 | 517.39 | 518.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 11:15:00 | 490.60 | 490.47 | 499.50 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-07 09:15:00 | 482.80 | 497.45 | 500.50 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 497.90 | 496.51 | 499.92 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-08 12:15:00 | 501.55 | 496.59 | 499.91 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 15:15:00 | 538.00 | 502.89 | 502.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 11:15:00 | 542.50 | 504.00 | 503.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 15:15:00 | 691.30 | 691.34 | 658.81 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-03 12:15:00 | 693.85 | 691.38 | 659.48 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 776.35 | 800.29 | 775.57 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-30 13:15:00 | 768.95 | 799.53 | 775.55 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 09:15:00 | 726.70 | 791.00 | 791.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 09:15:00 | 719.85 | 782.92 | 787.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 10:15:00 | 775.15 | 773.85 | 781.29 | EMA200 retest candle locked |

### Cycle 5 — BUY (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 10:15:00 | 838.90 | 786.75 | 786.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 11:15:00 | 840.00 | 787.28 | 786.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-12 10:15:00 | 808.20 | 810.02 | 800.27 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-14 09:15:00 | 818.20 | 810.27 | 801.01 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-20 14:15:00 | 799.45 | 812.59 | 803.42 | Close below EMA400 |

### Cycle 6 — SELL (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 13:15:00 | 772.60 | 796.34 | 796.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 14:15:00 | 764.55 | 796.02 | 796.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 12:15:00 | 793.40 | 779.66 | 786.30 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-23 14:15:00 | 766.20 | 782.90 | 787.32 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-10 09:15:00 | 703.75 | 667.21 | 702.34 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-11-08 09:15:00 | 210.55 | 2024-01-03 15:15:00 | 237.79 | TARGET | 27.24 |
| SELL | 2025-04-07 09:15:00 | 482.80 | 2025-04-08 12:15:00 | 501.55 | EXIT_EMA400 | -18.75 |
| BUY | 2025-07-03 12:15:00 | 693.85 | 2025-08-12 12:15:00 | 796.97 | TARGET | 103.12 |
| BUY | 2026-01-14 09:15:00 | 818.20 | 2026-01-20 14:15:00 | 799.45 | EXIT_EMA400 | -18.75 |
| SELL | 2026-02-23 14:15:00 | 766.20 | 2026-03-04 09:15:00 | 702.83 | TARGET | 63.37 |
