# SJVN Ltd. (SJVN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 78.98
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
- **Total realized P&L (per unit):** 5.00
- **Avg P&L per closed trade:** 1.25

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 12:15:00 | 133.63 | 137.92 | 137.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 09:15:00 | 133.20 | 137.75 | 137.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 14:15:00 | 134.65 | 132.04 | 134.20 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-30 09:15:00 | 131.30 | 132.05 | 134.18 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 12:15:00 | 133.20 | 132.06 | 134.15 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-09-30 13:15:00 | 131.44 | 132.05 | 134.14 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-11-28 15:15:00 | 119.30 | 112.97 | 118.25 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 15:15:00 | 104.35 | 94.78 | 94.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 105.68 | 94.89 | 94.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 97.81 | 98.23 | 96.76 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-05 09:15:00 | 103.21 | 98.10 | 96.88 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 98.65 | 99.79 | 98.05 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-16 09:15:00 | 97.00 | 99.68 | 98.05 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 11:15:00 | 95.12 | 98.08 | 98.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 13:15:00 | 94.63 | 98.02 | 98.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 09:15:00 | 97.74 | 95.97 | 96.89 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-14 09:15:00 | 92.45 | 95.93 | 96.81 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-08-20 09:15:00 | 97.98 | 95.44 | 96.46 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 80.68 | 73.35 | 73.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 81.07 | 73.64 | 73.47 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-30 09:15:00 | 131.30 | 2024-10-07 09:15:00 | 122.66 | TARGET | 8.64 |
| SELL | 2024-09-30 13:15:00 | 131.44 | 2024-10-07 09:15:00 | 123.34 | TARGET | 8.10 |
| BUY | 2025-06-05 09:15:00 | 103.21 | 2025-06-16 09:15:00 | 97.00 | EXIT_EMA400 | -6.21 |
| SELL | 2025-08-14 09:15:00 | 92.45 | 2025-08-20 09:15:00 | 97.98 | EXIT_EMA400 | -5.53 |
