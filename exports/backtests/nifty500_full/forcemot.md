# Force Motors Ltd. (FORCEMOT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (4487 bars)
- **Last close:** 19904.00
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
- **Winners / losers:** 0 / 3
- **Target hits / EMA400 exits:** 0 / 3
- **Total realized P&L (per unit):** -2051.05
- **Avg P&L per closed trade:** -683.68

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 12:15:00 | 8214.20 | 8485.49 | 8486.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 14:15:00 | 8127.45 | 8443.14 | 8464.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 10:15:00 | 7604.00 | 7578.62 | 7915.54 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-03 13:15:00 | 7429.85 | 7571.26 | 7873.52 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-10-30 09:15:00 | 7661.90 | 6964.44 | 7358.00 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 09:15:00 | 7473.50 | 6733.39 | 6730.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-07 13:15:00 | 7574.95 | 6809.88 | 6770.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 11:15:00 | 18770.00 | 18853.03 | 17174.41 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-16 09:15:00 | 19332.00 | 18573.99 | 17408.25 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 17740.00 | 18605.81 | 17714.36 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-26 10:15:00 | 17703.00 | 18580.74 | 17715.00 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 14:15:00 | 16638.00 | 17198.17 | 17199.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-16 15:15:00 | 16530.00 | 17191.52 | 17195.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-17 09:15:00 | 17418.00 | 17193.78 | 17196.94 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 11:15:00 | 17646.00 | 17203.75 | 17201.92 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-10-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 13:15:00 | 16339.00 | 17196.75 | 17200.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 13:15:00 | 16282.00 | 17141.41 | 17171.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 14:15:00 | 17121.00 | 17030.77 | 17109.68 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 10:15:00 | 18332.00 | 17186.93 | 17183.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-06 09:15:00 | 18613.00 | 17249.94 | 17215.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 10:15:00 | 17408.00 | 17437.76 | 17326.15 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-14 11:15:00 | 17490.00 | 17423.51 | 17323.23 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-18 09:15:00 | 17300.00 | 17441.72 | 17338.51 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-03 13:15:00 | 7429.85 | 2024-10-30 09:15:00 | 7661.90 | EXIT_EMA400 | -232.05 |
| BUY | 2025-09-16 09:15:00 | 19332.00 | 2025-09-26 10:15:00 | 17703.00 | EXIT_EMA400 | -1629.00 |
| BUY | 2025-11-14 11:15:00 | 17490.00 | 2025-11-18 09:15:00 | 17300.00 | EXIT_EMA400 | -190.00 |
