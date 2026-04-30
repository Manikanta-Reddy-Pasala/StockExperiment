# Aditya Birla Sun Life AMC Ltd. (ABSLAMC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1013.25
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 1 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 1 / 2
- **Target hits / EMA400 exits:** 1 / 2
- **Total realized P&L (per unit):** 53.96
- **Avg P&L per closed trade:** 17.99

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 14:15:00 | 755.75 | 795.21 | 795.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 10:15:00 | 739.20 | 793.81 | 794.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 10:15:00 | 636.75 | 635.39 | 673.93 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-07 09:15:00 | 622.25 | 640.66 | 664.79 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-21 11:15:00 | 659.50 | 636.05 | 656.71 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 10:15:00 | 716.80 | 662.71 | 662.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 13:15:00 | 722.00 | 672.83 | 667.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 14:15:00 | 736.25 | 741.50 | 714.76 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 09:15:00 | 748.75 | 741.47 | 715.01 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 833.75 | 855.00 | 828.88 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-02 14:15:00 | 827.60 | 853.24 | 829.14 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 09:15:00 | 809.70 | 824.63 | 824.69 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 10:15:00 | 854.25 | 824.62 | 824.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 11:15:00 | 860.90 | 824.98 | 824.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 14:15:00 | 830.45 | 837.53 | 831.80 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 09:15:00 | 773.15 | 826.57 | 826.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 10:15:00 | 771.10 | 826.02 | 826.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 753.70 | 745.27 | 768.55 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2026-01-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 15:15:00 | 855.00 | 777.40 | 777.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 09:15:00 | 862.00 | 797.54 | 791.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 859.40 | 859.99 | 832.16 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-05 09:15:00 | 886.90 | 860.31 | 833.28 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-30 10:15:00 | 876.90 | 911.02 | 877.84 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-04-07 09:15:00 | 622.25 | 2025-04-21 11:15:00 | 659.50 | EXIT_EMA400 | -37.25 |
| BUY | 2025-06-20 09:15:00 | 748.75 | 2025-07-10 12:15:00 | 849.96 | TARGET | 101.21 |
| BUY | 2026-03-05 09:15:00 | 886.90 | 2026-03-30 10:15:00 | 876.90 | EXIT_EMA400 | -10.00 |
