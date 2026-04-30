# Samvardhana Motherson International Ltd. (MOTHERSON.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 121.53
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 4 |
| ENTRY1 | 3 |
| ENTRY2 | 2 |
| EXIT | 3 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 3 / 2
- **Target hits / EMA400 exits:** 3 / 2
- **Total realized P&L (per unit):** 26.15
- **Avg P&L per closed trade:** 5.23

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 10:15:00 | 119.23 | 129.28 | 129.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 11:15:00 | 118.77 | 129.17 | 129.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 10:15:00 | 116.10 | 115.68 | 120.49 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-06 11:15:00 | 114.93 | 115.67 | 120.46 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 88.58 | 85.86 | 90.75 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-25 10:15:00 | 88.43 | 85.88 | 90.74 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 88.83 | 86.36 | 90.53 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-28 10:15:00 | 88.73 | 86.38 | 90.52 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 11:15:00 | 87.43 | 84.18 | 87.76 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-21 13:15:00 | 88.29 | 84.26 | 87.76 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 09:15:00 | 95.57 | 89.44 | 89.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 11:15:00 | 95.81 | 89.90 | 89.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 13:15:00 | 99.73 | 99.84 | 96.55 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-19 10:15:00 | 100.61 | 99.85 | 96.62 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-22 10:15:00 | 99.50 | 101.64 | 99.61 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 12:15:00 | 89.94 | 98.77 | 98.78 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 11:15:00 | 107.30 | 97.72 | 97.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 12:15:00 | 107.91 | 97.82 | 97.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 103.15 | 104.02 | 101.82 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-14 14:15:00 | 105.06 | 103.86 | 102.06 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 104.06 | 105.27 | 103.63 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-11-06 11:15:00 | 103.41 | 105.24 | 103.63 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 12:15:00 | 110.78 | 119.61 | 119.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 107.29 | 118.30 | 118.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 117.41 | 115.14 | 117.12 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 09:15:00 | 127.31 | 118.48 | 118.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 10:15:00 | 129.13 | 118.58 | 118.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 120.05 | 121.28 | 120.02 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-06 11:15:00 | 114.93 | 2025-01-10 09:15:00 | 98.34 | TARGET | 16.59 |
| SELL | 2025-03-25 10:15:00 | 88.43 | 2025-04-04 09:15:00 | 81.49 | TARGET | 6.94 |
| SELL | 2025-03-28 10:15:00 | 88.73 | 2025-04-04 09:15:00 | 83.35 | TARGET | 5.38 |
| BUY | 2025-06-19 10:15:00 | 100.61 | 2025-07-22 10:15:00 | 99.50 | EXIT_EMA400 | -1.11 |
| BUY | 2025-10-14 14:15:00 | 105.06 | 2025-11-06 11:15:00 | 103.41 | EXIT_EMA400 | -1.65 |
