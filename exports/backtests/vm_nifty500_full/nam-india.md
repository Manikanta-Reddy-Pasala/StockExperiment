# Nippon Life India Asset Management Ltd. (NAM-INDIA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 1009.95
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT3 | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** 38.71
- **Avg P&L per closed trade:** 9.68

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 10:15:00 | 683.20 | 707.37 | 707.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 10:15:00 | 665.35 | 705.66 | 706.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 09:15:00 | 549.45 | 549.41 | 587.01 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-07 09:15:00 | 528.15 | 565.20 | 583.62 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-15 12:15:00 | 580.20 | 560.32 | 578.29 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 09:15:00 | 634.55 | 591.29 | 591.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 09:15:00 | 646.85 | 596.78 | 594.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 10:15:00 | 721.40 | 722.20 | 683.11 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-13 14:15:00 | 735.00 | 722.41 | 683.98 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-28 14:15:00 | 796.50 | 824.31 | 798.46 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 13:15:00 | 821.65 | 856.17 | 856.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 14:15:00 | 813.95 | 853.41 | 854.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 12:15:00 | 855.60 | 851.26 | 853.67 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 11:15:00 | 900.65 | 855.76 | 855.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-18 12:15:00 | 904.00 | 856.24 | 855.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 14:15:00 | 864.05 | 866.60 | 861.72 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-31 12:15:00 | 880.45 | 867.07 | 862.41 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-12 09:15:00 | 861.45 | 875.78 | 868.40 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-01-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 14:15:00 | 797.40 | 863.89 | 864.16 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2026-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 12:15:00 | 927.30 | 864.00 | 863.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 936.00 | 866.60 | 865.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 921.40 | 924.38 | 902.69 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2026-03-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 14:15:00 | 832.20 | 888.00 | 888.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 812.55 | 879.58 | 883.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 10:15:00 | 876.85 | 873.46 | 880.12 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-25 11:15:00 | 863.95 | 873.37 | 880.04 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-08 09:15:00 | 895.00 | 859.11 | 870.71 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 10:15:00 | 971.30 | 880.91 | 880.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 986.35 | 886.07 | 883.14 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-04-07 09:15:00 | 528.15 | 2025-04-15 12:15:00 | 580.20 | EXIT_EMA400 | -52.05 |
| BUY | 2025-06-13 14:15:00 | 735.00 | 2025-08-28 14:15:00 | 796.50 | EXIT_EMA400 | 61.50 |
| BUY | 2025-12-31 12:15:00 | 880.45 | 2026-01-12 09:15:00 | 861.45 | EXIT_EMA400 | -19.00 |
| SELL | 2026-03-25 11:15:00 | 863.95 | 2026-03-30 09:15:00 | 815.69 | TARGET | 48.26 |
