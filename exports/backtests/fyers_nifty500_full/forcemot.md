# Force Motors Ltd. (FORCEMOT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 19980.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 2 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / EMA400 exits:** 2 / 2
- **Total realized P&L (per unit):** -315.96
- **Avg P&L per closed trade:** -78.99

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-02 14:15:00 | 9079.85 | 8693.75 | 8692.90 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-08-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 14:15:00 | 8550.05 | 8691.28 | 8691.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 13:15:00 | 8391.65 | 8683.11 | 8687.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 10:15:00 | 8536.35 | 8510.27 | 8585.19 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-27 12:15:00 | 8422.85 | 8544.62 | 8593.24 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 8335.85 | 8477.80 | 8552.16 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-09-04 12:15:00 | 8200.05 | 8448.07 | 8530.67 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-10-30 09:15:00 | 7661.90 | 6964.38 | 7374.35 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 15:15:00 | 7426.00 | 6723.76 | 6723.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 7473.50 | 6731.22 | 6727.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 11:15:00 | 18770.00 | 18853.41 | 17174.74 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-16 09:15:00 | 19332.00 | 18574.78 | 17408.85 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 17735.00 | 18605.08 | 17714.20 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-26 10:15:00 | 17703.00 | 18579.81 | 17714.73 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 14:15:00 | 16638.00 | 17198.00 | 17199.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-16 15:15:00 | 16530.00 | 17191.35 | 17195.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-17 09:15:00 | 17418.00 | 17193.61 | 17196.94 | EMA200 retest candle locked |

### Cycle 5 — BUY (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 11:15:00 | 17648.00 | 17203.70 | 17201.98 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-10-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 13:15:00 | 16339.00 | 17196.13 | 17199.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 13:15:00 | 16282.00 | 17140.94 | 17171.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 14:15:00 | 17121.00 | 17029.71 | 17109.15 | EMA200 retest candle locked |

### Cycle 7 — BUY (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 10:15:00 | 18332.00 | 17186.32 | 17183.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-06 09:15:00 | 18613.00 | 17249.31 | 17215.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 10:15:00 | 17408.00 | 17437.48 | 17325.94 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-14 11:15:00 | 17490.00 | 17423.54 | 17323.18 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-18 09:15:00 | 17300.00 | 17441.81 | 17338.49 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-08-27 12:15:00 | 8422.85 | 2024-09-06 09:15:00 | 7911.67 | TARGET | 511.18 |
| SELL | 2024-09-04 12:15:00 | 8200.05 | 2024-09-10 13:15:00 | 7208.19 | TARGET | 991.86 |
| BUY | 2025-09-16 09:15:00 | 19332.00 | 2025-09-26 10:15:00 | 17703.00 | EXIT_EMA400 | -1629.00 |
| BUY | 2025-11-14 11:15:00 | 17490.00 | 2025-11-18 09:15:00 | 17300.00 | EXIT_EMA400 | -190.00 |
