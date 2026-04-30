# Sai Life Sciences Ltd. (SAILIFE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-12-18 09:15:00 → 2026-04-30 15:15:00 (2361 bars)
- **Last close:** 1053.10
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 3 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / EMA400 exits:** 2 / 2
- **Total realized P&L (per unit):** 25.81
- **Avg P&L per closed trade:** 6.45

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 10:15:00 | 736.80 | 702.19 | 702.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-26 10:15:00 | 742.15 | 706.33 | 704.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 13:15:00 | 713.20 | 721.57 | 713.14 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-15 10:15:00 | 725.40 | 707.94 | 707.04 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-04-22 12:15:00 | 709.95 | 714.07 | 710.53 | Close below EMA400 |

### Cycle 2 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 818.50 | 886.11 | 886.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 800.10 | 885.25 | 885.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 870.05 | 866.71 | 875.41 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-04 09:15:00 | 855.60 | 867.19 | 875.36 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 855.60 | 867.19 | 875.36 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-04 12:15:00 | 830.50 | 866.44 | 874.86 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 840.10 | 860.64 | 871.42 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-09 09:15:00 | 887.45 | 860.10 | 870.78 | Close above EMA400 |

### Cycle 3 — BUY (started 2026-02-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 13:15:00 | 920.00 | 879.61 | 879.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 09:15:00 | 928.05 | 880.90 | 880.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 11:15:00 | 951.95 | 957.14 | 927.78 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-13 15:15:00 | 969.95 | 957.01 | 928.29 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 949.10 | 973.83 | 947.78 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-04-06 09:15:00 | 938.35 | 972.68 | 948.09 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-04-15 10:15:00 | 725.40 | 2025-04-15 14:15:00 | 780.49 | TARGET | 55.09 |
| SELL | 2026-02-04 09:15:00 | 855.60 | 2026-02-05 09:15:00 | 796.33 | TARGET | 59.27 |
| SELL | 2026-02-04 12:15:00 | 830.50 | 2026-02-09 09:15:00 | 887.45 | EXIT_EMA400 | -56.95 |
| BUY | 2026-03-13 15:15:00 | 969.95 | 2026-04-06 09:15:00 | 938.35 | EXIT_EMA400 | -31.60 |
