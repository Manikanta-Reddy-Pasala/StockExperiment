# SJVN Ltd. (SJVN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 79.06
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT3 | 2 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / EMA400 exits:** 2 / 2
- **Total realized P&L (per unit):** 4.23
- **Avg P&L per closed trade:** 1.06

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 10:15:00 | 133.16 | 137.68 | 137.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 14:15:00 | 132.87 | 137.49 | 137.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 14:15:00 | 134.75 | 132.03 | 134.12 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-30 09:15:00 | 131.30 | 132.03 | 134.10 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 12:15:00 | 133.18 | 132.04 | 134.07 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-09-30 13:15:00 | 131.53 | 132.04 | 134.06 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-11-28 15:15:00 | 119.26 | 112.94 | 118.18 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 14:15:00 | 104.20 | 94.69 | 94.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 105.68 | 94.89 | 94.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 97.71 | 98.23 | 96.74 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-05 09:15:00 | 103.21 | 98.10 | 96.86 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 98.65 | 99.78 | 98.03 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-16 09:15:00 | 97.00 | 99.68 | 98.04 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 11:15:00 | 95.15 | 98.09 | 98.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 93.86 | 97.98 | 98.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 09:15:00 | 97.83 | 95.97 | 96.89 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-14 09:15:00 | 92.45 | 95.93 | 96.81 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-08-20 09:15:00 | 97.98 | 95.44 | 96.46 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 79.55 | 73.30 | 73.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 10:15:00 | 80.84 | 73.38 | 73.31 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-30 09:15:00 | 131.30 | 2024-10-07 09:15:00 | 122.91 | TARGET | 8.39 |
| SELL | 2024-09-30 13:15:00 | 131.53 | 2024-10-07 09:15:00 | 123.95 | TARGET | 7.58 |
| BUY | 2025-06-05 09:15:00 | 103.21 | 2025-06-16 09:15:00 | 97.00 | EXIT_EMA400 | -6.21 |
| SELL | 2025-08-14 09:15:00 | 92.45 | 2025-08-20 09:15:00 | 97.98 | EXIT_EMA400 | -5.53 |
