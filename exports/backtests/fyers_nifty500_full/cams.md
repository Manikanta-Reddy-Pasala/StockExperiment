# Computer Age Management Services Ltd. (CAMS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 742.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 2 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** 35.15
- **Avg P&L per closed trade:** 8.79

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 10:15:00 | 884.60 | 949.80 | 950.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 09:15:00 | 865.98 | 945.85 | 948.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 699.06 | 695.37 | 756.33 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 10:15:00 | 787.20 | 758.59 | 758.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 11:15:00 | 791.52 | 758.91 | 758.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 794.42 | 803.90 | 785.64 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-16 12:15:00 | 807.12 | 803.15 | 786.14 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 812.84 | 829.98 | 811.99 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-11 11:15:00 | 807.66 | 829.76 | 811.96 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 09:15:00 | 750.46 | 808.67 | 808.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 09:15:00 | 743.04 | 796.23 | 802.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 12:15:00 | 772.68 | 772.09 | 783.78 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-08 13:15:00 | 769.52 | 772.07 | 783.71 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 780.82 | 772.47 | 782.97 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-11 12:15:00 | 776.90 | 772.70 | 782.93 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-09-17 09:15:00 | 782.00 | 772.72 | 781.72 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 11:15:00 | 805.00 | 778.07 | 778.03 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 751.50 | 778.44 | 778.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 746.00 | 777.38 | 778.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 11:15:00 | 767.70 | 764.28 | 770.21 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-23 11:15:00 | 760.90 | 764.18 | 769.95 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-24 09:15:00 | 771.10 | 764.09 | 769.76 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 10:15:00 | 774.20 | 699.61 | 699.33 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-06-16 12:15:00 | 807.12 | 2025-06-30 09:15:00 | 870.05 | TARGET | 62.93 |
| SELL | 2025-09-08 13:15:00 | 769.52 | 2025-09-17 09:15:00 | 782.00 | EXIT_EMA400 | -12.48 |
| SELL | 2025-09-11 12:15:00 | 776.90 | 2025-09-17 09:15:00 | 782.00 | EXIT_EMA400 | -5.10 |
| SELL | 2025-12-23 11:15:00 | 760.90 | 2025-12-24 09:15:00 | 771.10 | EXIT_EMA400 | -10.20 |
